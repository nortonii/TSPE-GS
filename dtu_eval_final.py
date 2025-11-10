import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    args = parser.parse_args()


script_dir = os.path.dirname(os.path.abspath(__file__))
for scene in dtu_scenes:
    scan_id = scene[4:]
    ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
    iteration = 30000
    string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
        f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
        f"--scan_id {scan_id} --output_dir {script_dir}/tmp/scan{scan_id} " + \
        f"--mask_dir {args.dtu} " + \
        f"--DTU {args.DTU_Official}"
    
    os.system(string)
#        f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \