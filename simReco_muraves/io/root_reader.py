from __future__ import annotations

import uproot

from simReco_muraves.models.raw_hit import RawHit


class RootReader:

    def __init__(
        self,
        filename: str,
        tree_name: str = "HitData",
    ):

        self.filename = filename
        self.tree_name = tree_name

        self.file = uproot.open(filename)
        self.tree = self.file[tree_name]

        self._arrays = None

    def _load_arrays(self):

        if self._arrays is None:

            self._arrays = self.tree.arrays(
                [
                    "ProcessID",
                    "Event",
                    "ScintHitStation",
                    "ScintHitModule",
                    "ScintHitBar",
                    "ScintHitE",
                    "ScintHitTime",
                    "ScintHitTrackID",
                    "ScintHitParentID",
                    "ScintHitPDG",
                ],
                library="np",
            )


    def read_event(
        self,
        process_id: int,
        event_id: int,
    ):

        self._load_arrays()

        arrays = self._arrays

        mask = (
            (arrays["ProcessID"] == process_id)
            &
            (arrays["Event"] == event_id)
        )

        hits = []

        indices = mask.nonzero()[0]

        for idx in indices:

            hits.append(
                RawHit(
                    process_id=int(arrays["ProcessID"][idx]),
                    event=int(arrays["Event"][idx]),
                    station=int(
                        arrays["ScintHitStation"][idx]
                    ),
                    module=int(
                        arrays["ScintHitModule"][idx]
                    ),
                    bar=int(
                        arrays["ScintHitBar"][idx]
                    ),
                    edep_mev=float(
                        arrays["ScintHitE"][idx]
                    ),
                    time_ns=float(
                        arrays["ScintHitTime"][idx]
                    ),
                    track_id=int(
                        arrays["ScintHitTrackID"][idx]
                    ),
                    parent_id=int(
                        arrays["ScintHitParentID"][idx]
                    ),
                    pdg=int(
                        arrays["ScintHitPDG"][idx]
                    ),
                )
            )

        return hits


    def get_event_ids(self):

        self._load_arrays()

        return sorted(
            set(
                self._arrays["Event"].tolist()
            )
        )

    def get_n_events(self):

        return len(
            self.get_event_keys()
        )


    def get_event_keys(self):

        self._load_arrays()

        process_ids = (
            self._arrays["ProcessID"]
        )

        events = (
            self._arrays["Event"]
        )

        return sorted(
            set(
                zip(
                    process_ids.tolist(),
                    events.tolist(),
                )
            )
        )
