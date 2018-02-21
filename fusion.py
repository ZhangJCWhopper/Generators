baseline = "foo"
mine = "10sec_gen_foo"
target = "10sec_new_foo"
with open(target, "w") as ot, open(baseline) as 1in, open(mine) as 2in:
	for 1line in 1in:
		2line = 2in.readline()
		1c = 1line.split()
		2c = 2line.split()
		if 1c[0] != 2c[0]:
			print("fatal error")
			exit(1)
		score = str((float(1c[1]) + float(2c[1])) *0.5)
		ot.write(1c[0] + " " + score+"\n")


