import matplotlib.pyplot as plt

file_values_avg_cf = [
    0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.0375, 0.05, 0.0625,
    0.075, 0.0875, 0.1,
]
avg_cf_values = [
    0.019878542020410284, 0.01981898128038533, 0.01662260473123666,
    0.016453567085553997, 0.016399169883124454, 0.01625924428334452,
    0.016186620541140236, 0.018100446919898337, 0.017083869310537635,
    0.01817646137334366, 0.020011187983695736
]

file_values_value3 = [
    0.0015625, 0.003125, 0.00625, 0.0125, 0.025,
    0.05, 0.0625, 0.075, 0.0875, 0.1,
]
value_3 = [
    0.003070489947952477, 0.002804776446968486, 0.0028060038755934385,
    0.0028014829248746596, 0.002804850696299533, 0.0027948172781527706,
    0.002801339271461524, 0.0028002543957950264, 0.002801146142144527,
    0.0027952771435609137,
]
plt.rcParams['font.family'] = 'Arial'

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300, facecolor="#EFE9E6")
ax1.set_facecolor("#EFE9E6")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(ls="--", lw=0.5, color="#4E616C")

color1 = "#3CB371"
ax1.set_xlabel("Hyperparameter Coefficient", fontsize=16)
ax1.set_ylabel("Chamfer Distance (Transparent Scenario)", color=color1, fontsize=18)
ax1.tick_params(axis='y', labelcolor=color1)
line1, = ax1.plot(
    file_values_avg_cf, avg_cf_values,
    color=color1, linestyle='-', linewidth=3,
    marker="s", mfc="white", ms=5,
    label='Transparent Scenario'
)

ax2 = ax1.twinx()
color2 = "#00579C"
ax2.set_ylabel("Chamfer Distance (Opaque Scenario)", color=color2, fontsize=18)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.spines["top"].set_visible(False)
line2, = ax2.plot(
    file_values_value3, value_3,
    color=color2, linestyle='dashdot', linewidth=3,
    marker="o", mfc="white", ms=5,
    label='Opaque Scenario'
)

# === Legend position control ===
# You can modify these two variables to adjust legend position:
legend_loc = 'upper right'  # e.g. 'upper right', 'upper left', 'center left', ...
legend_bbox = (0.6, 1)          # e.g. (1.02, 1), (0.5, 0.5), or None to ignore 

if legend_bbox is not None:
    ax1.legend(handles=[line1, line2], loc=legend_loc, bbox_to_anchor=legend_bbox, frameon=False, fontsize=20)
else:
    ax1.legend(handles=[line1, line2], loc=legend_loc, frameon=False, fontsize=20)

fig.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("test_dual_axis_plot.png", dpi=300, facecolor=fig.get_facecolor())