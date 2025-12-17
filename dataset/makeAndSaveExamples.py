import dataMaker as dtmk
import os

dataN = "Data/Noisy"
dataC = "Data/Clean"
#Make folders for data
os.makedirs(dataN, exist_ok=True)
os.makedirs(dataC, exist_ok=True)

#Make Noisy Sin data for train and val
dtmk.makeSinPT(100_000, random_state=1337,
               noise = 0.1, num_gaps=3,
               filename= dataN + "/Sin")
#Make clean sin data for testing
dtmk.makeSinPT(100_000, random_state=42,
               noise = 0.0, num_gaps=3,
               filename= dataC + "/Sin")

#Make Noisy CrazySin data for train and val
dtmk.makeCrazySinPT(100_000, random_state=1337,
                    noise = 0.1,
                    filename= dataN + "/CrazySin")
#Make Clean CrazySin data for testing
dtmk.makeCrazySinPT(100_000, random_state=42,
                    noise = 0.0,
                    filename= dataC + "/CrazySin")

#Make Noisy Friedman data for train and val
dtmk.makeFriedmanPT(100_000, random_state=1337,
                    noise = 0.1,
                    filename= dataN + "/Friedman")
#Make Clean Friedman data for testing
dtmk.makeFriedmanPT(100_000, random_state=42,
                    noise = 0.0,
                    filename= dataC + "/Friedman")