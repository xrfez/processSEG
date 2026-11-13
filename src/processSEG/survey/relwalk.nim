## A field record's relations station by station: the one walk of parseSEG's
## relations that `qc/fold`, `qc/record` and `survey/patch` share. Internal
## to processSEG (the umbrella does not re-export this module).
##
## parseSEG's `relationRuns` gives a record's relations as runs, each a range
## of evenly spaced stations on one line with `components` consecutive
## channels per station. Walking them costs one station key per station,
## where `relations` computes one per channel and checks which run owns it.
## The runs give exactly what `relations` gives when they ascend without
## sharing a channel (`relationsDisjoint`) and cover as many channels as
## parseSEG counts for the record (`expectedChannels`). The second test
## catches a multi-component run whose last channel is not the last
## component of a station: `relations` also gives the channels after it,
## on a station past the run's last point. Any other record is walked with
## `relations`, one channel at a time.

import parseSEG

type
  StationChannels* = object
    ## `channels` consecutive channels of a record, from `channel`, all
    ## recorded from the stations of `relation`: the components of one
    ## multi-component station, else one channel.
    channel*: int64
    channels*: int
    relation*: StationRelation ## The first channel's (`component` 1, or 0 without components).

func coveredByRuns(g: SurveyGeometry, record: int64, domain: RelationDomain): bool =
  # The runs alone give every channel of `record` once, as `relations` does.
  if g.relationsDisjoint(record, domain):
    var covered = 0'i64
    for run in g.relationRuns(record, domain):
      covered += int64(run.stations) * int64(run.components)
    result = covered == int64(g.expectedChannels(record, domain))

iterator stationChannels*(g: SurveyGeometry, record: int64,
    domain = rdSource): StationChannels =
  ## The channels of `record` (X; an O gather with `rdReceiver`) with their
  ## stations, in the order `relations` gives them, grouped by station.
  if g.coveredByRuns(record, domain):
    for run in g.relationRuns(record, domain):
      let component = if run.components > 1: 1 else: 0
      for i in 0 ..< run.stations:
        let moving = run.station(i)
        var rel = StationRelation(source: run.fixed, receiver: moving, component: component)
        if domain == rdReceiver: (rel.source, rel.receiver) = (moving, run.fixed)
        yield StationChannels(channel: run.fromChannel + int64(i) * run.channelStep,
          channels: run.components, relation: rel)
  else:
    for (channel, rel) in g.relations(record, domain):
      yield StationChannels(channel: channel, channels: 1, relation: rel)

iterator channelRelations*(g: SurveyGeometry, record: int64,
    domain = rdSource): tuple[channel: int64, relation: StationRelation] =
  ## What parseSEG's `relations` yields, channel by channel, from
  ## `stationChannels`.
  for s in g.stationChannels(record, domain):
    var rel = s.relation
    for k in 0 ..< s.channels:
      if k > 0: rel.component = s.relation.component + k
      yield (s.channel + int64(k), rel)
