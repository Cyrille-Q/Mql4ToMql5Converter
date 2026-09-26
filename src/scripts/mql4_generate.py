import random

functions = [
    "iMA(NULL,0,{p},0,MODE_SMA,PRICE_CLOSE,0)",
    "iRSI(NULL,0,{p},PRICE_CLOSE,0)",
    "iATR(NULL,0,{p},0)",
    "iCCI(NULL,0,{p},PRICE_CLOSE,0)"
]

orders = [
    "OrderSend(Symbol(),OP_BUY,{lot},Ask,3,0,0,\\\"Buy\\\",0,0,clrGreen);",
    "OrderSend(Symbol(),OP_SELL,{lot},Bid,3,0,0,\\\"Sell\\\",0,0,clrRed);"
]

conditions = [
    "if(Close[0] > Open[0])",
    "if(Close[0] < Open[0])",
    "if(Ask > Bid)",
    "if(Volume[0] > 100)"
]

def generate_code():
    period = random.choice([5,10,14,20,50,100])
    lot = round(random.uniform(0.01,1.0),2)
    func = random.choice(functions).format(p=period)
    cond = random.choice(conditions)
    order = random.choice(orders).format(lot=lot)

    code = f"""
int start(){{
 double val = {func};
 {cond}{{
  {order}
 }}
 return 0;
}}
"""
    return code.strip().replace("\n","\\n") #.replace("\"","\\\"")

with open("mql4_dataset.jsonl","w") as f:
    for _ in range(5000):
        code = generate_code()
        line = f'{{"text":"{code}"}}\n'
        f.write(line)
