from simReco_muraves.io.root_reader import RootReader


reader = RootReader(
    "/Users/sbasnet/Documents/MURAVES_stuff/digiWork/srcDigi/dataFiles/simData/musimData_run4.root"
)

hits = reader.read_event(0)

print(
    f"Event 0: {len(hits)} raw hits"
)

for hit in hits[:10]:

    print(
        hit
    )
