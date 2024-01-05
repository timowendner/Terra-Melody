import torch
from mido import MidiFile
from torch import nn


def get_midi(model: nn.Module, length: int):
    return
    model.eval()
    sequence = torch.zeros((1, length+1, 6)).float()

    for i in range(1, length+1):
        x = sequence[:, :i, :]
        with torch.no_grad():
            output = model(x, [i])

        sequence[:, i, :] = output[:, -1]

    # midi_stream = stream.Stream()
    # for note_index in sequence:
    #     # Convert the note index back to a MIDI note
    #     midi_note = note.Note(note_index)

    #     # Append the note to the MIDI stream
    #     midi_stream.append(midi_note)

    # # Write the MIDI stream to a file
    # midi_filename = "generated_midi.mid"
    # midi_stream.write('midi', fp=midi_filename)


def open_midi_file(path: str) -> list[tuple[float]]:
    """Get the start, duration, difference to the last starting-point,
    velocity and pitch of every single note in the midi file.

    Args:
        path (str): path to the midi file

    Returns:
        list[tuple[float]]: list of tuples of 
            - float: note staring point
            - float: note difference to the last starting-point
            - float: note duration
            - float: note velocity
            - float: note key
            - float: note octave
    """
    midi_file = MidiFile(path)
    result = [(0,)*6]
    open_notes = {}
    current = 0
    last = 0
    for msg in midi_file.tracks[0]:
        current += msg.time
        if msg.type == 'note_on':
            pitch = msg.note
            open_notes[pitch] = (current, current - last, msg.velocity)
            last = current
        elif msg.type == 'note_off' and (pitch := msg.note) in open_notes:
            start, diff, velocity = open_notes[pitch]

            speed = 50_000
            start, diff = start / speed, diff / speed
            octave, key = divmod(pitch, 12)
            del open_notes[pitch]
            duration = current/speed - start
            result.append(
                (start, diff, duration, velocity/60, octave, key)
            )
            # result.append((start, diff, current/speed - start, velocity/60))
    return result


class CombinedLoss(nn.Module):
    """Loss function for the Torch network."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        # error = cross_entropy(predictions[:, 14:, :], targets[:, 14:, :])
        # error += cross_entropy(predictions[:, 4:14, :], targets[:, 4:14, :])
        # error += mse(predictions[:, :4, :], targets[:, :4, :])
        error = mse(predictions, targets)
        return error
