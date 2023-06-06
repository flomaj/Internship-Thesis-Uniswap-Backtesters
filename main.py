import sys
from Strategies.Strategy1 import strategy1
from Strategies.Strategy2 import strategy2
from Strategies.Strategy3 import strategy3

def main(strat, token):
    if strat == "1":
        if token == "BLUR":
            strategy1("Data/dataBLUR.csv")
        elif token == "APE":
            strategy1("Data/dataAPE.csv")
        elif token == "HOP":
            strategy1("Data/dataHOP.csv")
        elif token == "PSP":
            strategy1("Data/dataPSP.csv")
        else:
            print("Invalid token.")
    elif strat == "2":
        if token == "BLUR":
            strategy2("Data/dataBLUR.csv")
        elif token == "APE":
            strategy2("Data/dataAPE.csv")
        elif token == "HOP":
            strategy2("Data/dataHOP.csv")
        elif token == "PSP":
            strategy2("Data/dataPSP.csv")
        else:
            print("Invalid token.")
    elif strat == "3":
        if token == "BLUR":
            strategy3("Data/dataBLUR.csv")
        elif token == "APE":
            strategy3("Data/dataAPE.csv")
        elif token == "HOP":
            strategy3("Data/dataHOP.csv")
        elif token == "PSP":
            strategy3("Data/dataPSP.csv")
        else:
            print("Invalid token.")
    else:
        print("Invalid strategy.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please provide two arguments: strat and token.")
        print("Usage: python main.py <strat> <token>")
    else:
        strat = sys.argv[1]
        token = sys.argv[2]
        main(strat, token)
