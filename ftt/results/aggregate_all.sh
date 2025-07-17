python3 -m ftt.results.aggregate \
    --exp_dir out/static/ \
    --parse_rules '{
        "nlayer": "lambda x: int(x.split(\"-\")[-1])",
        "repid": "lambda x: int(x.split(\"-\")[0])"
    }'

python3 -m ftt.results.aggregate \
    --exp_dir out/adaptive \
    --parse_rules '{"rho": "lambda x: float(x.split(\"-\")[-5])", "approach": "lambda x: x.split(\"-\")[-3]", "interval": "lambda x: int(x.split(\"-\")[-1])", "repid": "lambda x: int(x.split(\"-\")[0])"}'

python3 -m ftt.results.aggregate \
    --exp_dir out/stochastic \
    --parse_rules '{"savings": "lambda x: float(x.split(\"-\")[-3])", "repid": "lambda x: int(x.split(\"-\")[0])"}'

python3 -m ftt.results.aggregate \
    --exp_dir out/green_trainer \
    --parse_rules '{"rho": "lambda x: float(x.split(\"-\")[-3])", "repid": "lambda x: int(x.split(\"-\")[0])"}'

python3 -m ftt.results.aggregate \
    --exp_dir out/bandits \
    --parse_rules '{"repid": "lambda x: int(x.split(\"-\")[0])", "bandit": "lambda x: x.split(\"-\")[1].split(\"=\")[1]", "gamma": "lambda x: float(x.split(\"-\")[-5].split(\"=\")[1])", "lmda": "lambda x: float(x.split(\"-\")[-4].split(\"=\")[1])", "delta": "lambda x: float(x.split(\"-\")[-3].split(\"=\")[1])", "sigma": "lambda x: float(x.split(\"-\")[-2].split(\"=\")[1])", "beta": "lambda x: float(x.split(\"-\")[-1].split(\"=\")[1])"}'
