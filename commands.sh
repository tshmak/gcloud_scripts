# Some gcloud commands
gcloud projects list                        # List available projects
gcloud config set project <project_name>    # Change project
gcloud config list                          # Examine config
gcloud compute instances list               # See whether any clusters are started
gsutil list                                 # List storage buckets
gcloud compute ssh <clustername-m>          # ssh into master node of clustername
gcloud compute ssh testcluster-m --zone asia-east1-a
sudo su # to be root and install packages
/opt/conda/bin/conda install scikit-learn

# Broad's datacluster set-up tool -- A wapper of gcloud tool
# https://github.com/Nealelab/cloudtools
cluster start <name> [args]
cluster submit <name> [args]
cluster connect <name> [args]
cluster diagnose <name> [args]
cluster stop <name>
cluster list
# example
cluster start testcluster -p 6
cluster start <clustername> --master-machine-type n1-standard-2 --num-workers 0            # Start a minimal cluster
cluster submit testcluster myhailscript.py
cluster submit testcluster myhailscript.py --args "arg1 arg2"
cluster connect <clustername> nb            # Start a jupyter notebook connection to the cluster
# use notebook - highly recommond for testing
cluster connect testcluster notebook
# Monitoring Hail jobs
cluster connect testcluster ui
