import argparse

from libreface import get_facial_attributes

def main_func():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", type=str, required=True, help = "Path to the video or image which you want to process through libreface")
    parser.add_argument("--output_path", type=str, default="sample_results.csv", help="Path to the csv where results should be saved. Defaults to 'sample_results.csv'")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use while inference. Can be 'cpu', 'cuda:0', 'cuda:1', ... Defaults to 'cpu'")
    parser.add_argument("--temp", type=str, default="./tmp", help="Path where the temporary results for facial attributes can be saved.")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of frames to process in a single batch when doing inference on a video.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers to be used in the dataloader while doing inference on a video.")

    args = parser.parse_args()

    get_facial_attributes(args.input_path, 
                          output_save_path=args.output_path, 
                          model_choice="joint_au_detection_intensity_estimator",
                          temp_dir=args.temp, 
                          device=args.device,
                          batch_size=args.batch_size,
                          num_workers=args.num_workers)

if __name__ ==  "__main__":
    main_func()
