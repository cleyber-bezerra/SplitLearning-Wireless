#!/bin/bash
#../../ns3 clean
../../ns3 configure
../../ns3 build
../../ns3 run scratch/SplitLearning-NS3/my_wifi_ap_net_rand.cc
echo "Network simulation complete!"
exit