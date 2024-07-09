#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/energy-module.h"
#include "ns3/propagation-module.h"
#include "ns3/propagation-loss-model.h"

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip> // Include the header for std::fixed and std::setprecision

using namespace ns3;
using namespace ns3::energy;

NS_LOG_COMPONENT_DEFINE("WirelessNetworkExample");

int main(int argc, char *argv[])
{
    // Set the locale to support Unicode output in the console
    std::setlocale(LC_ALL, "");

    // Set a random seed
    RngSeedManager::SetSeed(1); // fetch current time: time(0) for Change the seed based on current time
    RngSeedManager::SetRun(1);  // Ensure same seed to guarantee same results each execution

    // Simulation parameters
    uint32_t nWifi = 10;
    double simulationTime = 10.0; // seconds
    double txPower = 50.0;         // dBm     (10, 30, 50)
    double stApDistance = 50.0;    // meters  (10, 50, 100)
    double lossExponent = 3; 	   // dB 2, 3, 4 for Low, moderate $ high exponents for low loss, moderate loss $ high losses
    double referenceLoss = 40.0;   // dB Varies between 40 and 100 or higher, 40dB for Wifi environment with minimal obstacles and low interference
    
    // Command line arguments
    CommandLine cmd;
    cmd.AddValue("nWifi", "Number of wifi STA devices", nWifi);
    cmd.AddValue("simulationTime", "Simulation time in seconds", simulationTime);
    cmd.Parse(argc, argv);

    // Create nodes
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(nWifi);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    // Map to store node ID mappings
    std::map<Ptr<Node>, uint32_t> nodeIdMap;

    // Configure WiFi Channel
    YansWifiPhyHelper phy = YansWifiPhyHelper();
    phy.SetPcapDataLinkType(YansWifiPhyHelper::DLT_IEEE802_11_RADIO); // for wireshark ns-3 RadioTap and Prism tracing extensions for 802.11

    YansWifiChannelHelper channel = YansWifiChannelHelper();
    channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    channel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent", DoubleValue(lossExponent),	
                               "ReferenceLoss", DoubleValue(referenceLoss));
    phy.SetChannel(channel.Create());

    // Set transmission power and standard
    phy.Set("TxPowerStart", DoubleValue(txPower)); // dBm
    phy.Set("TxPowerEnd", DoubleValue(txPower));   // dBm
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);

    // Install WiFi to all nodes
    WifiMacHelper mac;
    Ssid ssid = Ssid("ns-3-ssid");
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices;
    staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevices;
    apDevices = wifi.Install(phy, mac, wifiApNode);

    // Set mobility
    MobilityHelper mobility; // Declare the mobility variable

    // Set initial positions in a grid
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(stApDistance),
                                  "DeltaY", DoubleValue(stApDistance),
                                  "GridWidth", UintegerValue(nWifi),
                                  "LayoutType", StringValue("RowFirst"));

    //mobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel", "Bounds", RectangleValue(Rectangle(-25, 25, -25, 25)));
    mobility.Install(wifiStaNodes);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(wifiApNode);

    // Install Internet stack
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);

    // Assign fixed IDs to nodes
    for (uint32_t i = 0; i < nWifi; ++i)
    {
        nodeIdMap[wifiStaNodes.Get(i)] = i + 1;
    }
    nodeIdMap[wifiApNode.Get(0)] = nWifi + 1;

    // Assign IP addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.3.0", "255.255.255.0");
    Ipv4InterfaceContainer staInterfaces;
    staInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer apInterface;
    apInterface = address.Assign(apDevices);

    // Install applications
    UdpEchoServerHelper echoServer(9);
    ApplicationContainer serverApp = echoServer.Install(wifiApNode.Get(0));
    serverApp.Start(Seconds(1.0));
    serverApp.Stop(Seconds(simulationTime + 1));

    // Random traffic generator using OnOffApplication
    ApplicationContainer clientApp;
    OnOffHelper onOff("ns3::UdpSocketFactory", Address(InetSocketAddress(apInterface.GetAddress(0), 9)));
    onOff.SetAttribute("DataRate", StringValue("50Mbps"));
    onOff.SetAttribute("PacketSize", UintegerValue(1024));
    onOff.SetAttribute("OnTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.5]"));
    onOff.SetAttribute("OffTime", StringValue("ns3::ExponentialRandomVariable[Mean=0.2]"));
    onOff.SetAttribute("MaxBytes", UintegerValue(1 * 1024 * 1024)); // Set maximum bytes to 1MB

    for (uint32_t i = 0; i < nWifi; ++i)
    {
        clientApp.Add(onOff.Install(wifiStaNodes.Get(i)));
    }

    clientApp.Start(Seconds(2.0));
    clientApp.Stop(Seconds(simulationTime + 1));

    // Energy model configuration
    LiIonEnergySourceHelper liIonSourceHelper;
    liIonSourceHelper.Set("LiIonEnergySourceInitialEnergyJ", DoubleValue(100.0)); // Initial energy of 100 J
    liIonSourceHelper.Set("InitialCellVoltage", DoubleValue(3.7));                // Initial cell voltage

    EnergySourceContainer sources = liIonSourceHelper.Install(wifiStaNodes);

    WifiRadioEnergyModelHelper radioEnergyHelper;
    DeviceEnergyModelContainer deviceModels = radioEnergyHelper.Install(staDevices, sources);

    // Collect initial energy levels
    std::vector<double> initialEnergies(nWifi);
    for (uint32_t j = 0; j < wifiStaNodes.GetN(); ++j)
    {
        Ptr<LiIonEnergySource> liIonSourcePtr = DynamicCast<LiIonEnergySource>(sources.Get(j));
        initialEnergies[j] = liIonSourcePtr->GetRemainingEnergy();
    }

    // Flow monitor
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();

    // Run simulation
    Simulator::Stop(Seconds(simulationTime + 2));
    Simulator::Run();

    // Print results
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    // Output statistics
    std::cout << "\nNETWORK PERFORMANCE STATISTICS:" << std::endl;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);

        // Check if it's a transmission to the server
        if (t.destinationAddress == apInterface.GetAddress(0))
        {
            uint32_t clientId = 0;
            double energyConsumed = 0.0;

            // Find the corresponding node index based on the source address
            for (auto &nodePair : nodeIdMap)
            {
                Ptr<Ipv4> ipv4 = nodePair.first->GetObject<Ipv4>();
                Ipv4Address addr = ipv4->GetAddress(1, 0).GetLocal();
                if (addr == t.sourceAddress)
                {
                    clientId = nodePair.second;

                    // Calculate energy consumption for transmission device
                    Ptr<LiIonEnergySource> liIonSourcePtr = DynamicCast<LiIonEnergySource>(sources.Get(clientId - 1));
                    energyConsumed = initialEnergies[clientId - 1] - liIonSourcePtr->GetRemainingEnergy();

                    break;
                }
            }

            std::cout << "\nClient: " << clientId << " (Flow ID " << i->first << ")"
                      << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")" << std::endl;
            std::cout << "Tx Packets = " << i->second.txPackets << std::endl;
            std::cout << "Rx Packets = " << i->second.rxPackets << std::endl;
            std::cout << "Throughput = " << std::fixed << std::setprecision(6) << i->second.rxBytes * 8.0 / (simulationTime - 1) / 1024 / 1024 << " Mbps" << std::endl;
            std::cout << "Latency = " << std::fixed << std::setprecision(6) << (i->second.rxPackets > 0 ? i->second.delaySum.GetSeconds() / i->second.rxPackets : INFINITY) << " s" << std::endl;
            std::cout << "Packet Loss Ratio = " << std::fixed << std::setprecision(6) << (i->second.txPackets > 0 ? (i->second.txPackets - i->second.rxPackets) * 100.0 / i->second.txPackets : INFINITY) << " %" << std::endl;
            std::cout << "Transmission Time = " << std::fixed << std::setprecision(6) << (i->second.txPackets > 0 ? (i->second.timeLastRxPacket - i->second.timeFirstTxPacket).GetSeconds() : INFINITY) << " s" << std::endl;
            std::cout << "Energy Consumed = " << std::fixed << std::setprecision(6) << energyConsumed << " J" << std::endl;
        }
    }

    // Output statistics to CSV file
    // std::cout << "\nWriting statistics to CSV file..." << std::endl;
    std::ofstream outputFile("./scratch/SplitLearning-NS3/csv/ns3/simulator_ns3.csv");
    outputFile << "Client,Flow ID,Source Address,Destination Address,Tx Packets,Rx Packets,Throughput (Mbps),Latency (s),Packet Loss Ratio (%),Transmission Time (s),Energy Consumed (J)\n";

    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);

        // Check if it's a transmission to the server
        if (t.destinationAddress == apInterface.GetAddress(0))
        {
            uint32_t clientId = 0;
            double energyConsumed = 0.0;

            // Find the corresponding node index based on the source address
            for (auto &nodePair : nodeIdMap)
            {
                Ptr<Ipv4> ipv4 = nodePair.first->GetObject<Ipv4>();
                Ipv4Address addr = ipv4->GetAddress(1, 0).GetLocal();
                if (addr == t.sourceAddress)
                {
                    clientId = nodePair.second;

                    // Calculate energy consumption for transmission device
                    Ptr<LiIonEnergySource> liIonSourcePtr = DynamicCast<LiIonEnergySource>(sources.Get(clientId - 1));
                    energyConsumed = initialEnergies[clientId - 1] - liIonSourcePtr->GetRemainingEnergy();

                    break;
                }
            }

            outputFile << clientId << "," << i->first << "," << t.sourceAddress << "," << t.destinationAddress << ","
                       << i->second.txPackets << "," << i->second.rxPackets << ","
                       << std::fixed << std::setprecision(6) << i->second.rxBytes * 8.0 / (simulationTime - 1) / 1024 / 1024 << ","
                       << std::fixed << std::setprecision(6) << (i->second.rxPackets > 0 ? i->second.delaySum.GetSeconds() / i->second.rxPackets : INFINITY) << ","
                       << std::fixed << std::setprecision(6) << (i->second.txPackets > 0 ? (i->second.txPackets - i->second.rxPackets) * 100.0 / i->second.txPackets : INFINITY) << ","
                       << std::fixed << std::setprecision(6) << (i->second.txPackets > 0 ? (i->second.timeLastRxPacket - i->second.timeFirstTxPacket).GetSeconds() : INFINITY) << ","
                       << std::fixed << std::setprecision(6) << energyConsumed << "\n";
        }
    }

    outputFile.close();
    std::cout << "\nStatistics written to CSV file.\n" << std::endl;

    // Cleanup
    Simulator::Destroy();

    return 0;
}

