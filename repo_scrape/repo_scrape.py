
import os
import wget


base_url = "http://archive.ubuntu.com/ubuntu/dists/bionic-updates/"

repo_list = ["main", "multiverse", "restricted", "universe"]
repo_files = ["Packages.gz", "Packages.xz", "Release"]

# get the base files for the repositories
# everything will be written from here on in the same  repo structure
wget.download(base_url + "Release", "Release")
wget.download(base_url + "Release.gpg", "Release.gpg")
wget.download(base_url + "Contents-amd64.gz", "Contents-amd64.gz")

# cycle through each of the repo_list folders and grab the files

print("")

for r in repo_list:
    print("Downloading from: " + base_url + r)

    dir_name = r + "/binary-amd64"
    try:
        os.makedirs(dir_name)

    except FileExistsError:
        print("")

    for f in repo_files:
        fn = r + "/binary-amd64/" + f
        print(fn)
        wget.download(base_url + fn, dir_name)

#    print(r + "/binary-amd64/Packages.xz")
#    wget.download(base_url + r + "/binary-amd64/Packages.xz", dir_name)
#    wget.download(base_url + + "/binary-amd64/Release", dir_name)

print("Complete")


