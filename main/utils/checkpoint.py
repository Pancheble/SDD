"""
utils/checkpoint.py
체크포인트 저장 / 로드 유틸리티.
"""
import os
import glob
import torch


def save_checkpoint(
    path: str,
    step: int,
    student,
    ema_student,
    proj_student,
    ema_proj_teacher,
    optimizer,
    scheduler,
    dino_loss,
    cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "student_state": student.state_dict(),
            "ema_state": ema_student.state_dict(),
            "proj_student_state": proj_student.state_dict(),
            "ema_proj_teacher_state": ema_proj_teacher.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "dino_center": dino_loss.dino_loss.center,
            "cfg": cfg,
        },
        path,
    )


def load_checkpoint(path: str, student, ema_student, proj_student,
                    ema_proj_teacher, optimizer, scheduler, dino_loss, device):
    ckpt = torch.load(path, map_location=device)
    student.load_state_dict(ckpt["student_state"])
    ema_student.load_state_dict(ckpt["ema_state"])
    proj_student.load_state_dict(ckpt["proj_student_state"])
    ema_proj_teacher.load_state_dict(ckpt["ema_proj_teacher_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if "dino_center" in ckpt:
        dino_loss.dino_loss.center.copy_(ckpt["dino_center"])
    return ckpt["step"]


def get_latest_checkpoint(output_dir: str):
    pattern = os.path.join(output_dir, "ckpt_step_*.pth")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def cleanup_old_checkpoints(output_dir: str, keep_last_n: int = 3):
    pattern = os.path.join(output_dir, "ckpt_step_*.pth")
    files = sorted(glob.glob(pattern))
    for f in files[:-keep_last_n]:
        os.remove(f)
