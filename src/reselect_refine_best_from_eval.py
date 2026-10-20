import os
import re
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        type=str,
        default="/media/jiang/ssd/zly/Occlusion/outputs/eval_ours_k6_v24.csv",
        help="Path to eval CSV (default: v24 ours csv)",
    )
    ap.add_argument(
        "--images_root",
        type=str,
        default="/media/jiang/ssd/zly/Occlusion/outputs/ours_k6",
        help="Root directory containing per-scene output folders",
    )
    ap.add_argument(
        "--score_col",
        type=str,
        default="scalar50",
        help="Currently only scalar50 is parsed from the first 4 CSV columns",
    )
    ap.add_argument(
        "--keep_backup",
        action="store_true",
        help="If set, keep old refine_best.* as refine_best_old.* before overwriting",
    )
    args = ap.parse_args()

    csv_path = args.csv_path
    images_root = args.images_root

    # 只选原始 6 张 refine，不碰 refine_best / refine_best_eval
    pat = re.compile(r"^refine_\d+\.(jpg|png|jpeg)$", re.IGNORECASE)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(images_root):
        raise NotADirectoryError(f"images_root not found: {images_root}")

    # scene -> (score, image)
    best = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
        cols = header.split(",")

        # 这份脚本只依赖前四列：scene,image,scalar,scalar50
        if len(cols) < 4 or cols[0] != "scene" or cols[1] != "image":
            raise RuntimeError(f"CSV header format unexpected: {cols[:6]}")

        if args.score_col != "scalar50":
            print(f"[WARN] This robust parser only reads the 4th column as scalar50. Ignoring score_col={args.score_col}")

        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # 只切前 4 段，后面 porous_details / dict/list 里再多逗号都不管
            parts = line.split(",", 4)
            if len(parts) < 4:
                print(f"[SKIP] bad line {lineno}")
                continue

            scene = parts[0].strip()
            image = parts[1].strip()
            scalar50_str = parts[3].strip()

            if not pat.match(image):
                continue

            try:
                score = float(scalar50_str)
            except Exception:
                print(f"[SKIP] bad score at line {lineno}: {scalar50_str}")
                continue

            if scene not in best or score < best[scene][0]:
                best[scene] = (score, image)

    report_path = os.path.join(images_root, "_reselect_report_v24.csv")
    n_ok = 0
    n_miss = 0

    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("scene,image,scalar50\n")

        for scene in sorted(best.keys()):
            score, image = best[scene]
            scene_dir = os.path.join(images_root, scene)
            src = os.path.join(scene_dir, image)

            if not os.path.exists(src):
                print(f"[MISS] {src}")
                n_miss += 1
                continue

            ext = os.path.splitext(image)[1].lower()
            dst_main = os.path.join(scene_dir, f"refine_best{ext}")
            dst_eval = os.path.join(scene_dir, f"refine_best_eval{ext}")

            if args.keep_backup:
                for old_name in [f"refine_best{ext}", f"refine_best_eval{ext}"]:
                    old_path = os.path.join(scene_dir, old_name)
                    if os.path.exists(old_path):
                        backup_path = os.path.join(scene_dir, old_name.replace("refine_best", "refine_best_old"))
                        shutil.copy(old_path, backup_path)

            shutil.copy(src, dst_main)
            shutil.copy(src, dst_eval)

            rf.write(f"{scene},{image},{score:.6f}\n")
            print(f"[OK] {scene}: {image} -> {os.path.basename(dst_main)}  score={score:.6f}")
            n_ok += 1

    print("\nDone.")
    print(f"Selected best for {n_ok} scenes")
    if n_miss > 0:
        print(f"Missing source images for {n_miss} scenes")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()