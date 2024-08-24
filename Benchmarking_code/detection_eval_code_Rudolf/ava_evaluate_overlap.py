import os


def load_from_txt(file_path):
    # Initialize two lists to hold the numbers from the first and second columns
    start_times = []
    end_times = []

    # Open the file and process its contents
    with open(file_path, 'r') as file:
        # Skip the first line
        next(file)

        # Iterate through the remaining lines
        for line in file:
            # Split each line by whitespace and convert to floats
            columns = line.strip().split()
            start_times.append(float(columns[0]))
            end_times.append(float(columns[1]))


    return start_times, end_times


def compute_iou(start_A, end_A, start_B, end_B):
    # Calculate the intersection
    intersection = max(0, min(end_A, end_B) - max(start_A, start_B))

    # Calculate the union
    union = (end_A - start_A) + (end_B - start_B) - intersection

    # Calculate the IoU
    iou = intersection / union if union != 0 else 0

    return iou


def evaluate_overlap(start_times, end_times, start_times_pred, end_times_pred, min_iou_for_hit=0.1, allow_multi_hit=True):
    true_positives = 0
    false_positives = 0

    target_indice_hit = []

    for i, start_time_pred in enumerate(start_times_pred):
        end_time_pred = end_times_pred[i]
        found_match = False
        for j, start_time in enumerate(start_times):
            end_time = end_times[j]

            iou = compute_iou(start_time, end_time, start_time_pred, end_time_pred)

            if iou >= min_iou_for_hit:
                if not found_match:
                    true_positives += 1
                found_match = True
                if not j in target_indice_hit:
                    target_indice_hit.append(j)
                #break
        if not found_match:
            false_positives += 1

    if allow_multi_hit:
        true_positives = len(target_indice_hit)

    false_negatives = len(start_times) - true_positives  # len(start_times) is all positives

    #print("true_positives: {}".format(true_positives))
    #print("false_positives: {}".format(false_positives))
    #print("false_negatives: {}".format(false_negatives))

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall, true_positives, false_positives, false_negatives

    #print("precision: {}".format(precision))
    #print("recall: {}".format(recall))





if __name__ == '__main__':
    recordings = []
    wav_file_dir = 'data/for_ava/'
    for wav_file in os.listdir(wav_file_dir):
        if wav_file[-4:] == '.WAV':
            if not 'SOX' in wav_file:
                wav_file = wav_file.split('.')[0]
                recordings.append(wav_file)


    min_iou_values_for_hit = [0.1 * i for i in range(9)]

    for min_iou_for_hit in min_iou_values_for_hit:

        output_file = "segmentation_metric_results/precision_recall_results_" + str(min_iou_for_hit).replace('.', '_') + ".txt"

        with open(output_file, 'w') as file:
            for recording in recordings:
                file.write('---------------------------' + recording + "----------------------------\n")
                print("------------------ " + recording + " ----------------------------")
                file_path_gt = 'data/labeled_converted/' + recording + '.txt'
                start_times_gt, end_times_gt = load_from_txt(file_path_gt)

                """
                file.write("ava\n")
                print("-------------- ava -------------------------")
                file_path_ava = 'segmentation_results_from_ava/' + recording + '.txt'
                start_times_ava, end_times_ava = load_from_txt(file_path_ava)

                precision, recall = evaluate_overlap(start_times_gt, end_times_gt, start_times_ava, end_times_ava, min_iou_for_hit=min_iou_for_hit)
                file.write(f"precision: {precision}\n")
                file.write(f"recall: {recall}\n")
                print("precision: {}".format(precision))
                print("recall: {}".format(recall))
                """

                file.write("ours\n")
                file_path_our_pred = 'segmentation_results_our/formatted/' + recording + '.txt'
                start_times_our_pred, end_times_our_pred = load_from_txt(file_path_our_pred)

                precision, recall = evaluate_overlap(start_times_gt, end_times_gt, start_times_our_pred, end_times_our_pred, min_iou_for_hit=min_iou_for_hit)
                file.write(f"precision: {precision}\n")
                file.write(f"recall: {recall}\n")
                print("precision: {}".format(precision))
                print("recall: {}".format(recall))


                """
                file.write("mupet\n")
                file_path_mupet_pred = 'segmentation_results_mupet/formatted/' + recording + '.txt'
                start_times_mupet_pred, end_times_mupet_pred = load_from_txt(file_path_mupet_pred)

                precision, recall = evaluate_overlap(start_times_gt, end_times_gt, start_times_mupet_pred, end_times_mupet_pred, min_iou_for_hit=min_iou_for_hit)
                file.write(f"precision: {precision}\n")
                file.write(f"recall: {recall}\n")
                print("precision: {}".format(precision))
                print("recall: {}".format(recall))
                """

                #print(start_times_ava)
                #print(start_times_gt)
