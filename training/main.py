"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from src.data_utils import (TrainDataset,TestDataset, genSpoof_list, load_trial_records)
from src.defense_utils import (
    apply_resolved_defense_config,
    forward_with_defense,
    get_defense_kwargs,
    get_defense_samples,
)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from src.path_utils import WorkflowPaths, apply_path_overrides, resolve_workflow_paths
from src.utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


def load_training_config(
    config_path: str,
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    ssl_pretrained_path: Optional[str] = None,
    defense_config_path: Optional[str] = None,
    musan_path: Optional[str] = None,
    rir_path: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    with open(config_path, "r") as file:
        config = json.load(file)

    if dataset_root is not None:
        config["database_path"] = dataset_root
    if metadata_root is not None:
        config["metadata_path"] = metadata_root
    if ssl_pretrained_path is not None:
        config["model_config"]["ssl_pretrained_path"] = ssl_pretrained_path
    if musan_path is not None:
        config["musan_path"] = musan_path
    if rir_path is not None:
        config["rir_path"] = rir_path
    if batch_size is not None:
        config["batch_size"] = batch_size

    return apply_resolved_defense_config(
        config=config,
        config_path=config_path,
        defense_config_path=defense_config_path,
    )


def apply_training_path_fallbacks(
    config: Dict[str, Any],
    dataset_root: Optional[str] = None,
    metadata_root: Optional[str] = None,
    train_trial_file: Optional[str] = None,
    dev_trial_file: Optional[str] = None,
    eval_trial_file: Optional[str] = None,
    train_audio_root: Optional[str] = None,
    dev_audio_root: Optional[str] = None,
    eval_audio_root: Optional[str] = None,
) -> Dict[str, Any]:
    if dataset_root is None:
        inferred_dataset_root = _infer_dataset_root_from_audio_roots(
            train_audio_root=train_audio_root,
            dev_audio_root=dev_audio_root,
            eval_audio_root=eval_audio_root,
        )
        if inferred_dataset_root is not None:
            config["database_path"] = inferred_dataset_root

    if metadata_root is None:
        for trial_file in (train_trial_file, dev_trial_file, eval_trial_file):
            if trial_file is not None:
                config["metadata_path"] = str(
                    Path(trial_file).expanduser().resolve(strict=False).parent
                )
                break

    return config


def _infer_dataset_root_from_audio_roots(
    train_audio_root: Optional[str] = None,
    dev_audio_root: Optional[str] = None,
    eval_audio_root: Optional[str] = None,
) -> Optional[str]:
    for raw_path in (train_audio_root, dev_audio_root, eval_audio_root):
        if raw_path is None:
            continue
        audio_root = Path(raw_path).expanduser().resolve(strict=False)
        if audio_root.name in {"flac_T", "flac_D", "flac_E"}:
            return str(audio_root.parent)
        if audio_root.name == "flac_E_eval" and audio_root.parent.name == "eval_full":
            return str(audio_root.parent.parent)
        return str(audio_root.parent)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        help="Configuration file.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        help="Output directory for results.",
        default="./exp_result",
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--metadata-root",
        "--metadata_root",
        dest="metadata_root",
        default=None,
        help="Optional metadata root override.",
    )
    parser.add_argument(
        "--train-trial-file",
        "--train_trial_file",
        dest="train_trial_file",
        default=None,
        help="Optional explicit train TSV override.",
    )
    parser.add_argument(
        "--dev-trial-file",
        "--dev_trial_file",
        dest="dev_trial_file",
        default=None,
        help="Optional explicit dev TSV override.",
    )
    parser.add_argument(
        "--eval-trial-file",
        "--eval_trial_file",
        dest="eval_trial_file",
        default=None,
        help="Optional explicit eval TSV override.",
    )
    parser.add_argument(
        "--train-audio-root",
        "--train_audio_root",
        dest="train_audio_root",
        default=None,
        help="Optional explicit train audio directory override.",
    )
    parser.add_argument(
        "--dev-audio-root",
        "--dev_audio_root",
        dest="dev_audio_root",
        default=None,
        help="Optional explicit dev audio directory override.",
    )
    parser.add_argument(
        "--eval-audio-root",
        "--eval_audio_root",
        dest="eval_audio_root",
        default=None,
        help="Optional explicit eval audio directory override.",
    )
    parser.add_argument(
        "--ssl-pretrained-path",
        "--ssl_pretrained_path",
        dest="ssl_pretrained_path",
        default=None,
        help="Optional WavLM checkpoint override.",
    )
    parser.add_argument(
        "--defense-config",
        "--defense_config",
        dest="defense_config",
        default=None,
        help="Optional shared defense config JSON override.",
    )
    parser.add_argument(
        "--musan-path",
        "--musan_path",
        dest="musan_path",
        default=None,
        help="Optional MUSAN directory override.",
    )
    parser.add_argument(
        "--rir-path",
        "--rir_path",
        dest="rir_path",
        default=None,
        help="Optional RIR directory override.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--start-val-epoch",
        "--start_val_epoch",
        dest="start_val_epoch",
        type=int,
        default=10,
        help="Epoch index to start validation.",
    )
    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="Run model training.",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Skip model training.",
    )
    parser.add_argument(
        "--eval",
        dest="eval",
        action="store_true",
        help="Run final evaluation on the eval split.",
    )
    parser.add_argument(
        "--no-eval",
        dest="eval",
        action="store_false",
        help="Skip final evaluation on the eval split.",
    )
    parser.set_defaults(train=True, eval=False)
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Comment to describe the saved model.",
    )
    parser.add_argument(
        "--eval-model-weights",
        "--eval_model_weights",
        dest="eval_model_weights",
        type=str,
        default=None,
        help="Path to the model weight file for evaluation.",
    )
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    if not args.train and not args.eval:
        raise ValueError("Nothing to do: enable training and/or evaluation.")

    # load experiment configurations
    config = load_training_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        metadata_root=args.metadata_root,
        ssl_pretrained_path=args.ssl_pretrained_path,
        defense_config_path=args.defense_config,
        musan_path=args.musan_path,
        rir_path=args.rir_path,
        batch_size=args.batch_size,
    )
    config = apply_training_path_fallbacks(
        config=config,
        dataset_root=args.dataset_root,
        metadata_root=args.metadata_root,
        train_trial_file=args.train_trial_file,
        dev_trial_file=args.dev_trial_file,
        eval_trial_file=args.eval_trial_file,
        train_audio_root=args.train_audio_root,
        dev_audio_root=args.dev_audio_root,
        eval_audio_root=args.eval_audio_root,
    )
    workflow_paths = resolve_workflow_paths(
        config=config,
        output_dir=args.output_dir,
        model_weights_path=args.eval_model_weights,
        require_training_assets=args.train,
        require_dev_assets=args.train,
        require_eval_assets=args.eval,
        train_audio_root_override=args.train_audio_root,
        dev_audio_root_override=args.dev_audio_root,
        eval_audio_root_override=args.eval_audio_root,
        train_metadata_override=args.train_trial_file,
        dev_metadata_override=args.dev_trial_file,
        eval_metadata_override=args.eval_trial_file,
    )
    config = apply_path_overrides(config, workflow_paths)
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = workflow_paths.output_dir
    dev_trial_path = workflow_paths.dev_metadata
    # define model related paths
    model_tag = config["model_tag"]
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    defense_kwargs = get_defense_kwargs(config)
    defense_samples = get_defense_samples(config)

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        workflow_paths,
        args.seed,
        config,
        load_train=args.train,
        load_dev=args.train,
        load_eval=args.eval,
    )


    optimizer = None
    scheduler = None
    optimizer_swa = None
    if args.train:
        optim_config["steps_per_epoch"] = len(trn_loader)
        optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
        optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    no_improve = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    if args.train:
        # Training
        for epoch in range(config["num_epochs"]):
            print("training epoch{:03d}".format(epoch))

            running_loss = train_epoch(trn_loader, model, optimizer, device,
                                       scheduler, config)
            if epoch < args.start_val_epoch:
                print("DONE.\nLoss:{:.5f}. Skip validation step".format(running_loss))
                continue

            produce_evaluation_file(dev_loader, model, device,
                                    metric_path/"dev_score.txt", dev_trial_path,
                                    defense_kwargs=defense_kwargs,
                                    defense_samples=defense_samples)
            dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
                cm_scores_file=metric_path/"dev_score.txt",
                output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
                printout=False)
            print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
                running_loss, dev_eer, dev_dcf, dev_cllr))
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_dcf", dev_dcf, epoch)
            writer.add_scalar("dev_cllr", dev_cllr, epoch)

            best_dev_dcf = min(dev_dcf, best_dev_dcf)
            best_dev_cllr = min(dev_cllr, best_dev_cllr)
            if best_dev_eer >= dev_eer:
                print("best model find at epoch", epoch)
                best_dev_eer = dev_eer
                torch.save(model.state_dict(),
                    model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
                if os.path.islink(os.path.join(model_save_path, 'best_model.pth')):
                    os.unlink(os.path.join(model_save_path, 'best_model.pth'))
                os.symlink("epoch_{}_{:03.3f}.pth".format(epoch, dev_eer),
                        os.path.join(model_save_path, 'best_model.pth'))
                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                no_improve = 0
            else:
                no_improve += 1
            writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
            writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
            writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)
            if no_improve >= config["early_stop_epochs"]:
                break

    # evaluates pretrained model on the evaluation split
    if args.eval:
        eval_trial_path = workflow_paths.eval_metadata
        model_path = workflow_paths.model_weights_path or (model_save_path / "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded : {}".format(model_path))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path,
                                defense_kwargs=defense_kwargs,
                                defense_samples=defense_samples)

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_result.txt")
        print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))
        sys.exit(0)

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config, device=device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        workflow_paths: WorkflowPaths,
        seed: int,
        config: dict,
        load_train: bool = True,
        load_dev: bool = True,
        load_eval: bool = True) -> Tuple[
            Optional[torch.utils.data.DataLoader],
            Optional[torch.utils.data.DataLoader],
            Optional[torch.utils.data.DataLoader]]:
    """Make PyTorch DataLoaders for train / development / evaluation."""

    trn_loader = None
    dev_loader = None
    eval_loader = None

    if load_train:
        trn_database_path = workflow_paths.train_audio_root
        trn_list_path = workflow_paths.train_metadata
        d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path)
        print("no. training files:", len(file_train))

        train_set = TrainDataset(list_IDs=file_train,
                                               labels=d_label_trn,
                                               base_dir=trn_database_path,
                                               add_noise=str_to_bool(config["add_noise"]),
                                               musan_path=config.get("musan_path", "musan_data"),
                                               rir_path=config.get("rir_path", "RIR_data"))
        gen = torch.Generator()
        gen.manual_seed(seed)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                shuffle=True,
                                drop_last=True,
                                pin_memory=torch.cuda.is_available(),
                                worker_init_fn=seed_worker,
                                generator=gen)

    if load_dev:
        dev_database_path = workflow_paths.dev_audio_root
        dev_trial_path = workflow_paths.dev_metadata
        _, file_dev = genSpoof_list(dir_meta=dev_trial_path)
        print("no. dev files:", len(file_dev))

        dev_set = TestDataset(list_IDs=file_dev,
                                                base_dir=dev_database_path)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=torch.cuda.is_available())

    if load_eval:
        eval_database_path = workflow_paths.eval_audio_root
        eval_trial_path = workflow_paths.eval_metadata
        _, file_eval = genSpoof_list(dir_meta=eval_trial_path)
        print("no. evaluation files:", len(file_eval))

        eval_set = TestDataset(list_IDs=file_eval,
                                                base_dir=eval_database_path)
        eval_loader = DataLoader(eval_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=torch.cuda.is_available())

    return trn_loader, dev_loader, eval_loader

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str,
    defense_kwargs: Optional[Dict[str, Any]] = None,
    defense_samples: int = 1) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    defense_kwargs = defense_kwargs or {}
    trial_records = load_trial_records(trial_path)
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out, _ = forward_with_defense(
                model=model,
                wav=batch_x,
                defense_kwargs=defense_kwargs,
                defense_samples=defense_samples,
            )
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    if len(trial_records) != len(fname_list) or len(fname_list) != len(score_list):
        raise ValueError(
            "Trial record, utterance, and score counts must match for score writing."
        )

    text_list = []
    for record, fn, sco in zip(trial_records, fname_list, score_list):
        if record["utterance_id"] != fn:
            raise ValueError(
                "Utterance ordering mismatch while writing scores: "
                f"expected '{record['utterance_id']}', got '{fn}'."
            )
        text_list.append(
            f"{record['speaker_id']} {record['utterance_id']} {sco} {record['label_name']}"
        )
    
    with open(save_path, "w") as fh:
        fh.write("\n".join(text_list) + '\n')
    del text_list
    fh.close()
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    import torch.nn as nn

    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # Loss function
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    defense_kwargs = get_defense_kwargs(config)
    defense_samples = get_defense_samples(config)


    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        # Move to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)


        # Forward pass
        batch_out, _ = forward_with_defense(
            model=model,
            wav=batch_x,
            defense_kwargs=defense_kwargs,
            defense_samples=defense_samples,
        )

        # Compute loss
        batch_loss = criterion(batch_out, batch_y)

        # Backpropagation
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        # Learning rate scheduler
        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError(f"scheduler error, got:{scheduler}")

    running_loss /= num_total
    return running_loss

if __name__ == "__main__":
    main(parse_args())
