from types import SimpleNamespace

import pytest

pytest.importorskip("streaming")

from stable_finance.dataset.mds import Stream, configure_epoch_size


def test_configure_epoch_size_covers_steps_and_clears_stream_weights(monkeypatch):
    stream = SimpleNamespace(proportion=1.0, repeat=2, choose=3)
    dataset = SimpleNamespace(
        streams=[stream],
        samples_per_stream=[100],
        shuffle_seed=7,
        _parallel_rank_world=SimpleNamespace(num_ranks=3),
    )

    def apply_weights(streams, samples_per_stream, epoch_samples, shuffle_seed):
        assert streams == [stream]
        assert samples_per_stream == [100]
        assert epoch_samples == 240
        assert shuffle_seed == 7
        return 240

    monkeypatch.setattr(Stream, "apply_weights", apply_weights)
    configure_epoch_size(dataset, total_steps=50, batch_size=4)

    assert dataset.epoch_size == 240
    assert dataset.length == 80
    assert not hasattr(stream, "proportion")
    assert not hasattr(stream, "repeat")
    assert not hasattr(stream, "choose")


@pytest.mark.parametrize("total_steps,batch_size", [(0, 1), (1, 0)])
def test_configure_epoch_size_requires_positive_sizes(total_steps, batch_size):
    with pytest.raises(ValueError, match="must be positive"):
        configure_epoch_size(SimpleNamespace(), total_steps, batch_size)
