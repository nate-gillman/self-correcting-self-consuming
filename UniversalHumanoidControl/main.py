import subprocess

# main_task = [
#     "python", "scripts/eval_uhc.py",
#     "--cfg", "uhc_explicit",
#     "--epoch", "5000",
#     "--data", "sample_data/amass_copycat_take5_test_small.pkl",
#     "--mode", "stats"
# ]

# main_task = [
#     "python", "scripts/eval_uhc.py",
#     "--cfg", "uhc_implicit",
#     "--epoch", "19000",
#     "--data", "sample_data/amass_copycat_take5_test_small.pkl",
#     "--mode", "stats"
# ]

# main_task = [
#     "python", "scripts/eval_uhc.py",
#     "--cfg", "uhc_implicit_shape",
#     "--epoch", "4700",
#     "--data", "sample_data/amass_copycat_take5_test_small.pkl",
#     "--mode", "stats"
# ]

main_task = [
    "python", "amass_to_imitation.py"
]

subprocess.run(main_task)