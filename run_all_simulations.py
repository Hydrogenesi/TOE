import subprocess

masses = [1e30, 2.5e30, 5e30, 1e31]

for mass in masses:
    print(f"\n🔥 Igniting simulation for mass: {mass} kg")
    subprocess.run(["python", "Carolina_Connection", "--mass", str(mass)])