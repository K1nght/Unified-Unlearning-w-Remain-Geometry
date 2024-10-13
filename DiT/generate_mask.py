import os 
import torch 
import argparse

def calc_sparsity(tensor):
    # Count zero elements
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)

    # Total number of elements
    total_elements = tensor.numel()

    # Compute sparsity
    sparsity = num_zero_elements / total_elements
    return sparsity.item(), total_elements, num_zero_elements

def main(args):
    for cls in args.forget_class:
        mask_path = os.path.join(args.mask_path, str(cls))
        ff = os.path.join(mask_path, "forget_fisher.pt")
        rf = os.path.join(mask_path, "remain_fisher.pt")

        forget_fisher = torch.load(ff)
        remain_fisher = torch.load(rf)

        ths = args.thresholds

        for th in ths:
            total_cnt = 0 
            w_cnt = 0 
            gradients = {}
            for name in forget_fisher.keys():
                gradients[name] = 0
                try:
                    weight_saliency = (forget_fisher[name] + 1e-15) / (remain_fisher[name] + 1e-15)
                    w = weight_saliency >= th
                    w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
                    total_cnt += total_elements
                    w_cnt += w_num_zero_elements
                    gradients[name] = w
                    # if "y_embedder" in name:
                    #     print(f"{name} total:{total_elements} weight:{w_sparsity}")
                except: 
                    print(f"{name} {forget_fisher[name]}")

            print(f"Total sparsity th:{th} weight:{w_cnt/total_cnt*100}")
            torch.save(gradients, os.path.join(mask_path, f"fisher_{th}.pt"))

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-path", required=True, type=str, default='./mask',
                        help="Path to the saliency mask.")
    parser.add_argument("--forget-class", nargs='+', type=int, required=True,
                        help="class to forget")  
    parser.add_argument("--thresholds", nargs='+', type=float, default=[0.5, 1, 3, 5, 10],
                        help="the thresholds")
    args = parser.parse_args()
    main(args)