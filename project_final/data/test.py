import pretty_midi
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def extract_pitch_and_frequency(midi_file_path, num_notes=20):
    # List to store the pitch and frequency data in the same sequence
    sequence_data = []
    freq = []

    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Counter to track the number of sampled notes
    note_count = 0

    # Iterate through all instruments in the MIDI file
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            # Extract the pitch (note number) and frequency
            pitch = note.pitch
            frequency = pretty_midi.note_number_to_hz(pitch)

            # Append the note data to the sequence
            sequence_data.append({'note': pitch, 'frequency': frequency})
            freq.append(frequency)

            # Increment the note count
            note_count += 1

            # Break the loop if the desired number of notes is sampled
            if note_count == num_notes:
                break

        # Break the loop if the desired number of notes is sampled
        if note_count == num_notes:
            break

    return sequence_data, freq

def smooth_data(data, window_size=3):
    df = pd.DataFrame(data)
    smoothed_data = df.rolling(window=window_size).mean().iloc[window_size-1:].values.flatten()
    return smoothed_data

# Example usage
midi_file_path = '/project/Harmonious-Music-Generation/project_final/generated_music/BBN_a.mid'
num_notes_to_sample = 50
sequence_data = extract_pitch_and_frequency(midi_file_path, 49)
pitch_data1, frequency_data1 = extract_pitch_and_frequency(
    '/project/Harmonious-Music-Generation/project_final/generated_music/LSTM_s.mid', num_notes_to_sample)

smoothed_pitch_data = smooth_data(frequency_data1, 3)
print(list(smoothed_pitch_data))
print(len(smoothed_pitch_data))

print(sequence_data[1])
print(len(sequence_data[1]))


plt.figure(figsize=(100, 10))  # Set the width and height of the plot
plt.plot(sequence_data[1], 'o-', label='Pitch (BBN)')
# plt.plot(frequency_data, 's-', label='Frequency (Hz)')
plt.plot(smoothed_pitch_data, 'o-', label='Pitch (LSTM)')
plt.xlabel('Note Index')
plt.ylabel('Frequency of notes (Hz)')
plt.title('Frequency of MIDI Notes')
plt.legend()

# Display values at each point
for i, value in enumerate(sequence_data[1]):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

for i, value in enumerate(smoothed_pitch_data):
    plt.text(i, value, f'{value:.2f}', ha='center', va='top')

plt.show()
plt.savefig("plot.png")