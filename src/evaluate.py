import json
import os

from src.utils import score_generation, score_generation_by_type


def reformat_hypotheses(hypotheses, opt):
    if opt.dset_name == "clevr":
        coco_format = []
        change_coco_format = []
        unchange_coco_format = []
        for k, v in hypotheses.items():
            coco_format.append({
                "image_id": k,
                "caption": v[0]["sentence"]
            })
            if k.endswith("nc"):
                unchange_coco_format.append({
                    "image_id": k,
                    "caption": v[0]["sentence"]
                })
            else:
                change_coco_format.append({
                    "image_id": k,
                    "caption": v[0]["sentence"]
                })
                
        total_coco_path = os.path.join(opt.res_dir, "coco_pred_val.json")
        change_coco_path = os.path.join(opt.res_dir, "coco_change_pred_val.json")
        unchange_coco_path = os.path.join(opt.res_dir, "coco_unchange_pred_val.json")
        
        with open(total_coco_path, "wt") as f:
            json.dump(coco_format, f, indent=4)
        with open(change_coco_path, "wt") as f:
            json.dump(change_coco_format, f, indent=4)
        with open(unchange_coco_path, "wt") as f:
            json.dump(unchange_coco_format, f, indent=4)
        return total_coco_path, change_coco_path, unchange_coco_path
    
    else:
        coco_format = []
        for k, v in hypotheses.items():
            coco_format.append({
                "image_id": k,
                "caption": v[0]["sentence"]
            })
            
        total_coco_path = os.path.join(opt.res_dir, "coco_pred_val.json")
        
        with open(total_coco_path, "wt") as f:
            json.dump(coco_format, f, indent=4)
        return total_coco_path, None, None


def evaluate(opt, all_res, anno_file, type_file=None):
    print("Reformatting hypotheses...")
    coco_file, change_coco_file, unchange_coco_file = reformat_hypotheses(all_res, opt)
    
    print("Scoring generation...")
    if opt.dset_name == "clevr":
        total_results = score_generation(anno_file, coco_file)
        change_results = score_generation(anno_file, change_coco_file)
        unchange_results = score_generation(anno_file, unchange_coco_file)
        if type_file:
            type_results = score_generation_by_type(anno_file, change_coco_file, type_file)
        else:
            type_results = None
        
        return {
            "total_results": total_results,
            "change_results": change_results,
            "unchange_results": unchange_results,
            "type_results": type_results
        }
    else:
        total_results = score_generation(anno_file, coco_file)
        return {
            "total_results": total_results
        }


if __name__ == "__main__":
    # Example usage
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dset_name", type=str, required=True, help="Dataset name (e.g., clevr)")
    parser.add_argument("--res_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--anno_file", type=str, required=True, help="Path to the annotation file")
    parser.add_argument("--type_file", type=str, default=None, help="Path to the type file (optional)")
    args = parser.parse_args()
    
    with open(os.path.join(args.res_dir, "model_best_greedy_pred_val.json"), "r") as f:
        all_res = json.load(f)["results"]

    res = evaluate(args, all_res, args.anno_file, args.type_file)
    
    with open(os.path.join(args.res_dir, "coco_eval_results.json"), "wt") as f:
        json.dump(res, f, indent=4)
        print(f"Results saved to {os.path.join(args.res_dir, 'coco_eval_results.json')}")
        