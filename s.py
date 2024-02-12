import json
import matplotlib.pyplot as plt


# plot_dir = 'results/dec_experiments/wqmix_tiger_plots'
plot_dir = 'results/dec_experiments/wqmix_box_pushing_plots'

#with open('/home/elem/repos/MARL/pymarl/results/sacred/tiger/cw_qmix_env=4_adam_td_lambda/1/info.json', 'r') as f:
with open('/home/elem/repos/MARL/pymarl/results/sacred/box_pushing/cw_qmix_env=4_adam_td_lambda/1/info.json', 'r') as f:
    data = json.load(f)

for key,value in data.items():
    plt.figure()
    plt.plot(value)
    plt.title(key)
    plt.savefig(f'{plot_dir}/{key}.png')
    plt.close()