import libreface

## inference on single image and store results to a variable
detected_facial_attributes = libreface.get_facial_attributes_image(image_path = "sample_disfa.png",
                                                                   temp_dir = "./temp",
                                                                   device = "cuda")

## inference on a single image and save results in a csv file
libreface.save_facial_attributes_image(image_path = "sample_disfa.png",
                                       output_save_path = "sample_image_results.csv",
                                       temp_dir = "./temp",
                                       device = "cuda")

# ## inference on a video and store the results to a pandas dataframe
detected_facial_attributes_df = libreface.get_facial_attributes_video(video_path = "sample_disfa.avi",
                                                                      temp_dir = "./temp",
                                                                      device = "cuda")

## inference on a video and save the results framewise in a csv file
libreface.save_facial_attributes_video(video_path = "sample_disfa.avi",
                                       output_save_path = "sample_video_results.csv",
                                       temp_dir = "./temp",
                                       device = "cuda")

## inference on any image or video type and store results accordingly to a variable or save results
detected_facial_attributes = libreface.get_facial_attributes(file_path = "sample_disfa.avi",
                                                             output_save_path = "sample_results.csv",
                                                             temp_dir = "./temp",
                                                             device = "cuda")