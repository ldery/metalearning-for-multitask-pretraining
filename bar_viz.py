import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 8))

labels = ['Eq. Weights', 'Alt-2', 'Alt-5', 'WarmUD-2', 'WarmUD-5', 'Meta-1.0', 'Meta-2.0', 'Meta-5.0']
perfs = [0.419, 0.406, 0.384, 0.430, 0.433, 0.340, 0.352, 0.368]
err = [0.021, 0.019, 0.014, 0.010, 0.006, 0.025, 0.012, 0.023]
x = [2, 3.75, 4.25, 5.75, 6.25, 7.75, 8.25, 8.75]
width = 0.5

ax.bar([x[0]], [perfs[0]], width, yerr=[err[0]], edgecolor='k', color='tab:blue', label='Eq. Weights')
ax.bar(x[1:3], perfs[1:3], width, yerr=err[1:3], edgecolor='k', color='tab:orange', label='Alternating')
ax.bar(x[3:5], perfs[3:5], width, yerr=err[3:5], edgecolor='k', color='tab:green', label='Warm up and down')
ax.bar(x[5:], perfs[5:], width, yerr=err[5:], edgecolor='k', color='tab:red', label='Meta')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy under different Training Regimes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim((0.3, 0.5))
ax.legend()

fig.tight_layout()

plt.savefig('perf_under_training.png')
plt.show()


# labels = ["reptiles", "flowers", "insects", "people", "fruit_and_vegetables"]


# target_only = [(0.496, 0.029), (0.614, 0.023), (0.543, 0.049), (0.338, 0.019), (0.656, 0.062)]
# joint = [(0.410, 0.006), (0.567, 0.033), (0.547, 0.021), (0.332, 0.028), (0.655, 0.002)]
# pretr_w_meta = [(0.574, 0.019), (0.644, 0.012), (0.664, 0.006), (0.367, 0.012), (0.720, 0.017)]
# pretr_joint = [(0.579, 0.017), (0.661, 0.014), (0.670, 0.019), (0.377, 0.004), (0.743, 0.012)]


# x = np.arange(len(labels))  # the label locations
# width = 0.1  # the width of the bars

# fig, ax = plt.subplots(figsize=(16, 8))
# rects1 = ax.bar(x - (width/2 + width), [a[0] for a in target_only], width, yerr=[a[1] for a in target_only], color='tab:blue', label='Target Only')
# rects2 = ax.bar(x - width/2, [a[0] for a in joint], width, color='tab:orange', yerr=[a[1] for a in joint], label='Joint w Equalized Weighting')

# rects3 = ax.bar(x + (width/2 + width), [a[0] for a in pretr_w_meta], width, color='tab:cyan', yerr=[a[1] for a in pretr_w_meta], label='Pretrained w Meta-Weights')
# rects4 = ax.bar(x + width/2, [a[0] for a in pretr_joint], width, color='tab:red', yerr=[a[1] for a in pretr_joint], label='Pretrained w Equalized Weighting')


# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy under different Training Regimes')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_ylim((0.3, 0.8))
# ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height + 0.03),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# # autolabel(rects1)
# # autolabel(rects2)
# # autolabel(rects3)
# # autolabel(rects4)


# fig.tight_layout()

# plt.savefig('perf_w_meta.png')
# plt.show()
