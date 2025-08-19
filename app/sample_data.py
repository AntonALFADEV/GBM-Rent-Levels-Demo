import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
n = 220  # rækker i syntetisk historik

def make_synth_df(n):
    base_date = datetime(2023, 1, 1)
    rows = []
    for _ in range(n):
        unit_id = int(rng.integers(1, 175))
        m2 = int(rng.integers(35, 115))
        vaer = max(1, int(round(m2 / int(rng.integers(18, 35)))))
        etage = int(rng.integers(0, 9))
        altan = int(rng.integers(0, 2))
        elevator = int(rng.integers(0, 2))
        udsigt = rng.choice(["ingen", "gård", "gade", "hav"] , p=[0.4,0.35,0.2,0.05])
        postnr = int(rng.choice([2100, 2200, 2300, 2400, 2450, 2500]))
        energi = rng.choice(list("ABCDEFG"), p=[0.05,0.08,0.1,0.2,0.25,0.22,0.1])
        orient = rng.choice(["N","S","E","W","SE","SW"], p=[0.1,0.25,0.15,0.15,0.2,0.15])
        klik = max(1, int(rng.normal(12, 6)))

        dstart = base_date + timedelta(days=int(rng.integers(0, 560)))

        base_ppm2 = 190 + 3*etage + (30 if altan else 0) + (20 if elevator else 0) + (60 if udsigt=="hav" else 20 if udsigt in ["gård","gade"] else 0)
        hood_adj = {2100:20,2200:35,2300:25,2400:10,2450:5,2500:15}[postnr]
        energy_adj = {"A":25,"B":20,"C":15,"D":10,"E":5,"F":0,"G":-5}[energi]
        ppm2 = base_ppm2 + hood_adj + energy_adj
        listepris = int(round(ppm2 * m2 / 12 / 10)*10)

        z = 1.2 \
            - 0.00005 * (listepris - 9000) \
            + 0.03 * altan + 0.02 * elevator + 0.015 * etage \
            + 0.0015 * (klik-10) \
            + (0.06 if udsigt in ["hav","gård"] else 0.03 if udsigt=="gade" else 0.0) \
            + (0.03 if energi in ["A","B","C"] else 0.0) \
            + (0.02 if orient in ["S","SE","SW"] else 0.0)
        p30 = 1/(1+np.exp(-z))
        udlejet_30 = rng.random() < p30

        if udlejet_30:
            dage = max(3, int(round(20 - 8*(p30-0.5) + rng.normal(0,3))))
            faktisk_leje = int(round(listepris * (1 - max(0, rng.normal(0.005, 0.01))) / 10)*10)
            d_udlejet = dstart + timedelta(days=dage)
        else:
            dage = int(round(40 + 30*(1-p30) + rng.normal(0,5)))
            faktisk_leje = ""
            d_udlejet = None

        rows.append({
            "unit_id": unit_id,
            "dato_start": dstart.date().isoformat(),
            "dato_udlejet": "" if d_udlejet is None else d_udlejet.date().isoformat(),
            "listepris": listepris,
            "faktisk_leje": faktisk_leje,
            "dage_til_udlejning": "" if d_udlejet is None else dage,
            "udlejet_inden_30_dage": int(udlejet_30),
            "m2": m2,
            "vaerelser": vaer,
            "etage": etage,
            "altan": altan,
            "elevator": elevator,
            "udsigt": udsigt,
            "orientering": orient,
            "postnr": postnr,
            "energi_maerke": energi,
            "klik_pr_dag": klik
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = make_synth_df(n)
    df.to_csv("sample_data.csv", index=False)
    print("Skrev sample_data.csv med", len(df), "rækker.")
