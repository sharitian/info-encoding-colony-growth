from flask import Flask, render_template, request, redirect, session, url_for
import os, subprocess
from subprocess import Popen, PIPE, check_output
import growth
import shutil



path = os.getcwd()
app = Flask(__name__)
app.secret_key = "secret key"

@app.route("/", methods = ["GET", "POST"])
def home():
	if request.method == "POST":
		#print(request.form["radius"])
		#print(request.form["spacing"])
		#print(request.form["d1d2"])
		#print(request.form["b"])
		#print(request.form["growth"])

		if os.path.isdir("./final"):
			shutil.rmtree(os.getcwd() + "/final")

		if os.path.isdir("./initial"):
			shutil.rmtree(os.getcwd() + "/initial")

		if os.path.isdir("./plaintext"):
			shutil.rmtree(os.getcwd() + "/plaintext")

		if os.path.isdir("./cropped"):
			shutil.rmtree(os.getcwd() + "/cropped")

		if os.path.isfile("./Simulation_Parameters.txt"):
			os.remove(os.getcwd() + "/Simulation_Parameters.txt")

		session["radius"] = request.form["radius"]
		session["spacing"] = request.form["spacing"]
		session["d1d2"] = request.form["d1d2"]
		session["b"] = request.form["b"]
		session["growth"] = request.form["growth"]

		return redirect(url_for("command_server1", command = command_server1))
	elif "output" in session:
		return render_template("home.html", output = session["output"])
	return render_template("home.html", output = "")


def run_command(command):
	return subprocess.Popen(command, shell = True, stdout = subprocess.PIPE).stdout.read()


@app.route("/command0/<command>")
def command_server0(command):
	#output = run_command("python " + path + "/growth.py " + session["radius"] + " " + session["spacing"]+ " " + session["d1d2"]+ " " + session["b"]+ " " + session["growth"])
	#session["output"] = output 
	#return redirect(url_for("home"))
	return run_command("python " + path + "/growth.py " + session["radius"] + " " + session["spacing"]+ " " + session["d1d2"]+ " " + session["b"]+ " " + session["growth"])

@app.route("/command1/<command>")
def command_server1(command):
	output = run_command("python " + path + "/test.py " + session["radius"] + " " + session["spacing"]+ " " + session["d1d2"]+ " " + session["b"]+ " " + session["growth"])
	session["output"] = output 
	return redirect(url_for("home"))
	#return run_command("python" + path + "/test.py" + session["radius"] + " " + session["spacing"]+ " " + session["d1d2"]+ " " + session["b"]+ " " + session["growth"])


@app.route("/restart")
def restart():
	session.pop("radius", None)
	session.pop("spacing", None)
	session.pop("d1d2", None)
	session.pop("b", None)
	session.pop("growth", None)

	if os.path.isdir("./final"):
		shutil.rmtree(os.getcwd() + "/final")

	if os.path.isdir("./initial"):
		shutil.rmtree(os.getcwd() + "/initial")

	if os.path.isdir("./plaintext"):
		shutil.rmtree(os.getcwd() + "/plaintext")

	if os.path.isdir("./cropped"):
		shutil.rmtree(os.getcwd() + "/cropped")

	if os.path.isfile("./Simulation_Parameters.txt"):
		os.remove(os.getcwd() + "/Simulation_Parameters.txt")

	return redirect(url_for("home"))



if __name__ == "__main__":
	app.run(debug = True)