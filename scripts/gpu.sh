#!/bin/bash

# Written by Wim R.M. Cardoen on 05/30/2025
#  Use:  gpu.sh  lonepeak|kingspeak|notchpeak|granite|redwood
# Update: 11/03/2025 (Added redwood)


# 1.Check whether a clustername is provided
clustername=$1
if [ -z "$clustername" ] 
then
    printf "  Cluster name is NOT provided!\n"
    printf "  MUST be either: lonepeak, kingspeak, notchpeak, granite or redwood\n\n"
    exit
fi


# 2.Check whether the clustername is valid
case "$clustername" in
  lonepeak|kingspeak|notchpeak|granite|redwood)
    printf "  CLUSTER:%s\n"  "$clustername"
    printf "  Date   :%s\n\n"  "$(date)"
    ;;
  *)
    printf "  Invalid clustername:%s!\n"  "$clustername"
    printf "  MUST be either: lonepeak, kingspeak, notchpeak, granite or redwood\n\n" 
    exit
    ;;
esac


# 3.Find all Nodes in a cluster
lstNodes=$(scontrol -M $clustername show nodes | grep -o "NodeName=[^ ]*")


# 4.Find all the nodes with GPUs in the cluster
#   If GPUS are found on a node:
#      Give the node name + features
#      Give gpu-type (gres string) and ngpus
#      Give ncores and memory of the node
totnodes=0
totgpus=0
for el in $lstNodes
do
    nodeid=$(echo $el | cut -d"=" -f 2) 
    hasgpu=$(scontrol -M $clustername show node=$nodeid | grep 'gpu:')
    if [ "$hasgpu" ] 
    then
        printf "  GPUs found on node:%s\n"  "$nodeid"
        features=$(scontrol -M $clustername show node=$nodeid | grep ActiveFeatures= | cut -d"=" -f 2 )
        totnodes=$(( totnodes + 1 ))
        gputype=$(echo $hasgpu | cut -d":" -f 2 )
        numgpus=$(echo $hasgpu | cut -d":" -f 3 | cut -d"(" -f 1 )
        totgpus=$(( totgpus + numgpus ))
        # cores
        line=$(scontrol -M $clustername show node=$nodeid | grep CPUTot=)
        ncores=$(echo $line | cut -d" " -f 3 | cut -d"=" -f 2) 
        # memory
        line=$(scontrol -M $clustername show node=$nodeid | grep RealMemory=)
        mem=$(echo $line | cut -d" " -f 1 | cut -d"=" -f 2)
        printf "    features (node):%s\n"    "$features"
        printf "    gpu-type:%s\n"    "$gputype"
        printf "    #gpus   :%s\n"    "$numgpus"
        printf "    #cores  :%s\n"    "$ncores"
        printf "    mem (MB):%s\n\n"  "$mem"
    fi
done 

printf "\n  Summary for the %s cluster:\n"  "$clustername"
printf "    #Nodes with GPUs:%d\n" "$totnodes"
printf "    #GPUs:%d\n" "$totgpus"
