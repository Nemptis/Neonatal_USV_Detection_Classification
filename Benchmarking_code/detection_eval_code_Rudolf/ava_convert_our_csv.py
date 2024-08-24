import pandas as pd
import numpy as np
import os

from scipy.io import wavfile


def remove_dots(time_list):
    start_times = []
    end_times = []
    for start_time, end_time in time_list:
        start_time = str(start_time)
        end_time = str(end_time)
        start_time_new = start_time.replace('.', '')
        end_time_new = end_time.replace('.', '')

        start_time_new = float(start_time_new.replace(',', '.'))
        end_time_new = float(end_time_new.replace(',', '.'))

        start_times.append(start_time_new)
        end_times.append(end_time_new)

    return np.array(start_times), np.array(end_times)


def timescale_to_seconds(start_times, end_times, wav_data, sampling_rate):
    # Convert units of start time and end time
    # This calculates a factor f = 10^n with n in Z as big as possible under the constraint, that f * endtime <= length of audio data.
    div = (len(wav_data) / sampling_rate) / max(end_times)
    div_order = int(np.log10(div))
    if div < 1:
        div_order -= 1
    factor = 10 ** div_order

    return start_times*factor, end_times*factor





def load_from_txt(file_path):
    # Initialize two lists to hold the numbers from the first and second columns
    first_column = []
    second_column = []

    # Open the file and process its contents
    with open(file_path, 'r') as file:
        # Skip the first line
        next(file)

        # Iterate through the remaining lines
        for line in file:
            # Split each line by whitespace and convert to floats
            columns = line.strip().split()
            first_column.append(float(columns[0]))
            second_column.append(float(columns[1]))

    # Print the resulting lists
    #print("First column:", first_column)
    #print("Second column:", second_column)

    return zip(first_column, second_column)



def convert_our_csv(csv_path, wav_path, output_path):

    sampling_rate, wav_data = wavfile.read(wav_path)

    # Read the CSV file
    df = pd.read_csv(csv_path, sep=';', skipinitialspace=True)
    if '#' in df.columns:
        df = df.drop(columns=['#'])

    #print(df['category'])

    # Extract the starttime and endtime columns
    times = df[['starttime', 'endtime']]

    # Convert the result to a list of tuples (starttime, endtime)
    time_list = list(times.itertuples(index=False, name=None))

    #print(time_list)

    # Print the list
    start_times, end_times = remove_dots(time_list)

    start_times, end_times = timescale_to_seconds(start_times, end_times, wav_data, sampling_rate)

    start_times = list(start_times)
    end_times = list(end_times)

    with open(output_path, 'w') as file:
        # Write the header line
        file.write('# Onsets/offsets for '+ csv_path +'\n')

        # Iterate through both lists and write each pair to the file
        for first, second in zip(start_times, end_times):
            file.write(f'{first:.5f} {second:.5f}\n')


    #print(start_times)
    #print(end_times)



def convert_from_mupet(csv_path, output_path):
    df = pd.read_csv(csv_path, sep=',', skipinitialspace=True)

    start_times = df['Syllable start time (sec)']
    end_times = df['Syllable end time (sec)']

    start_times_float = []
    end_times_float = []

    for i, start_time in enumerate(start_times):
        start_times_float.append(float(start_time))
        end_times_float.append(float(end_times[i]))

    with open(output_path, 'w') as file:
        # Write the header line
        file.write('# Onsets/offsets for '+ csv_path +'\n')

        # Iterate through both lists and write each pair to the file
        for first, second in zip(start_times, end_times):
            file.write(f'{first:.5f} {second:.5f}\n')

if __name__ == '__main__':

    recordings = []
    wav_file_dir = 'data/for_ava/'
    for wav_file in os.listdir(wav_file_dir):
        if wav_file[-4:] == '.WAV':
            if not 'SOX' in wav_file:
                wav_file = wav_file.split('.')[0]
                recordings.append(wav_file)

    #recordings = ['glu397_9874_p8_2_0', '']

    for recording in recordings:
        # convert the ground truth data
        file_path = 'data/labeled/' + recording + '.csv'
        wav_path = 'data/labeled/' + recording + '.WAV'
        output_path = 'data/labeled_converted/' + recording + '.txt'
        convert_our_csv(file_path, wav_path, output_path=output_path)

        # convert the predicted data
        file_path = 'segmentation_results_our/raw/' + recording + '.csv'
        output_path = 'segmentation_results_our/formatted/' + recording + '.txt'
        convert_our_csv(file_path, wav_path, output_path=output_path)

        # convert the mupet predicted data
        file_path = 'segmentation_results_mupet/raw/' + recording + '.csv'
        output_path = 'segmentation_results_mupet/formatted/' + recording + '.txt'
        convert_from_mupet(file_path, output_path)

