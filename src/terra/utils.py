import torch
from mido import MidiFile
from torch import nn


def open_midi_file(path: str) -> list[tuple[float]]:
    """Get the start, duration, difference to the last starting-point,
    velocity and pitch of every single note in the midi file.

    Args:
        path (str): path to the midi file

    Returns:
        list[tuple[float]]: list of tuples of 
            - float: note staring point
            - float: note duration
            - float: note difference to the last starting-point
            - float: note velocity
            - 8 floats: one-hot encoding of key
            - 8 floats: one-hot encoding of octave
    """
    midi_file = MidiFile(path)
    result = [(0,)*26]
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
            octave, key = torch.eye(10)[octave], torch.eye(12)[key]
            del open_notes[pitch]
            result.append(
                (start, current/speed - start, diff, velocity, *key, *octave)
            )
    return sorted(result)


class CombinedLoss(nn.Module):
    """Loss function for the Torch network."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        error = cross_entropy(predictions[:, 15:, :], targets[:, 15:, :])
        error += cross_entropy(predictions[:, 5:15, :], targets[:, 5:15, :])
        error += mse(predictions[:, :5, :], targets[:, :5, :])
        return error
