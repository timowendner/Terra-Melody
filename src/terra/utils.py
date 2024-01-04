import torch
from mido import MidiFile
from torch import nn


def open_midi_file(path: str) -> list[tuple[float]]:
    """Get the start, duration, difference to the last starting-point,
    velocity and pitch of every single note in the midi file.

    Args:
        path (str): path to the midi file

    Returns:
        list[tuple[float]]: tuple of 
            - float: note staring point
            - float: note duration
            - float: note difference to the last starting-point
            - float: note velocity
            - float: note key
    """
    midi_file = MidiFile(path)
    result = []
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
            del open_notes[pitch]
            result.append((start, current - start, diff, velocity, pitch))
    return sorted(result)


class CombinedLoss(nn.Module):
    """Loss function for the Torch network."""

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = torch.mean(torch.abs(predictions - targets))
        return mse
