/*
 *  Create a synapse index map.
 *
 *  @param  synapseIndexMap   Reference to the pointer to SynapseIndexMap structure.
 *  @param  sim_info          Pointer to the simulation information.
 *  @param  clr_info          Pointer to the cluster information.
 */

#include "SynapseIndexMap.h"
#include "AllSynapses.h"
#include "Cluster.h"

/**
 *  Get cluster index from neuron layout index.
 *
 *  @param  iNeuron      Neuron layout index.
 *  @param  vtClrInfo    Vecttor of pointer to the ClusterInfo object.
 */
CLUSTER_INDEX_TYPE SynapseIndexMap::getClusterIdxFromNeuronLayoutIdx(int iNeuron, vector<ClusterInfo *> &vtClrInfo) {
    CLUSTER_INDEX_TYPE iCluster = 0;
    for (iCluster = 0; iCluster < vtClrInfo.size(); iCluster++) {
        if (iNeuron >= vtClrInfo[iCluster]->clusterNeuronsBegin &&
            iNeuron < vtClrInfo[iCluster]->clusterNeuronsBegin + vtClrInfo[iCluster]->totalClusterNeurons) {
            break;
        }
    }
    return iCluster;
}

/**
 *  Get internal neuron index in cluster from neuron layout index.
 *
 *  @param  iNeuron      Neuron layout index.
 *  @param  vtClrInfo    Vecttor of pointer to the ClusterInfo object.
 */
int SynapseIndexMap::getNeuronIdxFromNeuronLayoutIdx(int iNeuron, vector<ClusterInfo *> &vtClrInfo) {
    CLUSTER_INDEX_TYPE iCluster = SynapseIndexMap::getClusterIdxFromNeuronLayoutIdx(iNeuron, vtClrInfo);

    return (iNeuron - vtClrInfo[iCluster]->clusterNeuronsBegin);
}

/*
 *  Create a synapse index map.
 *
 *  @param  sim_info          Pointer to the simulation information.
 *  @param  vtClr             Vector of pointer to the Cluster object.
 *  @param  vtClrInfo         Vecttor of pointer to the ClusterInfo object.
 */
void SynapseIndexMap::createSynapseImap(const SimulationInfo* sim_info, vector<Cluster *> &vtClr, vector<ClusterInfo *> &vtClrInfo)
{
    // allocate memories for forward map
    int totalNeurons = sim_info->totalNeurons;
    vector<OUTGOING_SYNAPSE_INDEX_TYPE>* rgSynapseSynapseIndexMap = new vector<OUTGOING_SYNAPSE_INDEX_TYPE>[totalNeurons];

    // create incoming synapse index map
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) {
        int neuron_count = vtClrInfo[iCluster]->totalClusterNeurons;
        AllSynapses *synapses = dynamic_cast<AllSynapses*>(vtClr[iCluster]->m_synapses);
        SynapseIndexMap *synapseIndexMap = vtClr[iCluster]->m_synapseIndexMap;
        BGSIZE total_incoming_synapse_count = 0;

        // count the total synapses
        for ( int iNeuron = 0; iNeuron < neuron_count; iNeuron++ )
        {
            assert( static_cast<int>(synapses->synapse_counts[iNeuron]) < sim_info->maxSynapsesPerNeuron );
            total_incoming_synapse_count += synapses->synapse_counts[iNeuron];
        }

        DEBUG ( cout << "\nCluster: " << iCluster << " total_incoming_synapse_count: " << total_incoming_synapse_count << endl; )

        if ( total_incoming_synapse_count == 0 )
        {
            continue;
        }

        if (synapseIndexMap != NULL)
        {
            delete synapseIndexMap;
            synapseIndexMap = NULL;
        }

        BGSIZE syn_i = 0;
        BGSIZE n_inUse = 0;
        BGSIZE inter_cluster_synapse_count = 0; 

        // create incomming synapse index map
        vtClr[iCluster]->m_synapseIndexMap = new SynapseIndexMap(neuron_count, total_incoming_synapse_count);

        // for each destination neuron in the cluster
        for (int iNeuron = 0; iNeuron < neuron_count; iNeuron++)
        {
            BGSIZE synapse_count = 0;
            vtClr[iCluster]->m_synapseIndexMap->incomingSynapseBegin[iNeuron] = n_inUse;
            for ( int j = 0; j < sim_info->maxSynapsesPerNeuron; j++, syn_i++ )
            {
                if ( synapses->in_use[syn_i] == true )
                {
                    int srcNeuronLayoutIndex = synapses->sourceNeuronLayoutIndex[syn_i];
                    OUTGOING_SYNAPSE_INDEX_TYPE idxSynapse = SynapseIndexMap::getOutgoingSynapseIndex(iCluster, syn_i);

                    // save for outgoing synapse index map
                    rgSynapseSynapseIndexMap[srcNeuronLayoutIndex].push_back(idxSynapse);

                    vtClr[iCluster]->m_synapseIndexMap->incomingSynapseIndexMap[n_inUse] = syn_i;
                    n_inUse++;
                    synapse_count++;

                    DEBUG(
                    CLUSTER_INDEX_TYPE idxCluster = SynapseIndexMap::getClusterIdxFromNeuronLayoutIdx(srcNeuronLayoutIndex, vtClrInfo);
                    if (iCluster != idxCluster) {
                        inter_cluster_synapse_count++;
                    }
                    )
                }
            }
            assert( synapse_count == synapses->synapse_counts[iNeuron] );
            vtClr[iCluster]->m_synapseIndexMap->incomingSynapseCount[iNeuron] = synapse_count;
        }

        assert( total_incoming_synapse_count == n_inUse );
        synapses->total_synapse_counts = total_incoming_synapse_count;

        DEBUG ( cout << "Cluster: " << iCluster << " inter_cluster_synapse_count: " << inter_cluster_synapse_count << endl; )
    }

    // create outgoing synapse index maps
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) 
    {
        int neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;

        // for each neuron in the cluster
        BGSIZE total_outgoing_synapse_count = 0; 
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
        {
            total_outgoing_synapse_count += rgSynapseSynapseIndexMap[neuronLayoutIndex].size();
        }

        if (total_outgoing_synapse_count == 0) {
            continue;
        }

        // Number of incoming synapes and number of outgoing synapses are not always equal.
        // However for the growth connection model, each couple of neurons are connected 
        // bi-directionally. So the number of inter cluster synapses between two clusters 
        // should be equal, and therefore number of incoming synapes and number of outgoing 
        // synapses are equal.
        vtClr[iCluster]->m_synapseIndexMap->allocOutgoingSynapseIndexMap(total_outgoing_synapse_count);

        DEBUG( cout << "\nCluster: " << iCluster << " total_outgoing_synapse_count: " << total_outgoing_synapse_count << endl; )

        neuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin;
        BGSIZE syn_i = 0;
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, neuronLayoutIndex++)
        {
            vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseBegin[iNeuron] = syn_i;
            vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseCount[iNeuron] = rgSynapseSynapseIndexMap[neuronLayoutIndex].size();

            for ( BGSIZE j = 0; j < rgSynapseSynapseIndexMap[neuronLayoutIndex].size(); j++, syn_i++)
            {
                vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseIndexMap[syn_i] = rgSynapseSynapseIndexMap[neuronLayoutIndex][j];
            }
        }
    }

    // delete memories
    delete[] rgSynapseSynapseIndexMap;

    DEBUG_MID (
    // check the validity of the incoming index map of SynapseIndexMap 
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++)
    {
        // for each destination neuron in the cluster
        int dstNeuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin; 
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, dstNeuronLayoutIndex++)
        {
            // check the validity of each incoming synapse of the destinaion neuron
            // Verify that every synapse in the incoming index map has a valid
            // neuron destination index.
            BGSIZE beginIdx = vtClr[iCluster]->m_synapseIndexMap->incomingSynapseBegin[iNeuron]; 
            BGSIZE count = vtClr[iCluster]->m_synapseIndexMap->incomingSynapseCount[iNeuron];
            for (BGSIZE idxMap = beginIdx; idxMap < beginIdx + count; idxMap++)
            {
                // get synapse index
                BGSIZE inSynIdx = vtClr[iCluster]->m_synapseIndexMap->incomingSynapseIndexMap[idxMap];

                // get the pointer to the incoming synapse
                IAllSynapses *inSynapses = vtClr[iCluster]->m_synapses;
         
                // get the desination neuron index of the synapse
                int dstNeuronLayoutIndex2 = (dynamic_cast<AllSynapses*>(inSynapses))->destNeuronLayoutIndex[inSynIdx]; 
                // check the validity of source neuron index
                if (dstNeuronLayoutIndex != dstNeuronLayoutIndex2)
                {
                    cout << "\niCluster: " << iCluster << " dstNeuronLayoutIndex: " << dstNeuronLayoutIndex << endl;
                    cout <<  "dstNeuronLayoutIndex2: " << dstNeuronLayoutIndex2 << endl;
                    assert(false);
                }
            }
        }
    }

    // check the validity of the outgoing index map of SynapseIndexMap.
    // Verify that every synapse in the outgoing index map has a valid
    // neuron source index.
    for (CLUSTER_INDEX_TYPE iCluster = 0; iCluster < vtClr.size(); iCluster++) 
    {
        // for each source neuron in the cluster
        int srcNeuronLayoutIndex = vtClrInfo[iCluster]->clusterNeuronsBegin; 
        int totalClusterNeurons = vtClrInfo[iCluster]->totalClusterNeurons;
        for (int iNeuron = 0; iNeuron < totalClusterNeurons; iNeuron++, srcNeuronLayoutIndex++)
        {
            // check the validity of each outgoing synapse of the source neuron
            BGSIZE beginIdx = vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseBegin[iNeuron]; 
            BGSIZE count = vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseCount[iNeuron];
            for (BGSIZE idxMap = beginIdx; idxMap < beginIdx + count; idxMap++)
            {
                // get cluster and synapse index
                OUTGOING_SYNAPSE_INDEX_TYPE outIdx = vtClr[iCluster]->m_synapseIndexMap->outgoingSynapseIndexMap[idxMap];
                CLUSTER_INDEX_TYPE outCluster = SynapseIndexMap::getClusterIndex(outIdx);
                BGSIZE outSynIdx = SynapseIndexMap::getSynapseIndex(outIdx);

                // get the pointer to the outgoing synapse
                IAllSynapses *outSynapses = vtClr[outCluster]->m_synapses;
         
                // get the source neuron index of the synapse
                int srcNeuronLayoutIndex2 = (dynamic_cast<AllSynapses*>(outSynapses))->sourceNeuronLayoutIndex[outSynIdx]; 
                // check the validity of source neuron index
                if (srcNeuronLayoutIndex != srcNeuronLayoutIndex2)
                {
                    cout << "\niCluster: " << iCluster << " srcNeuronLayoutIndex: " << srcNeuronLayoutIndex << endl;
                    cout << "outCluster: " << outCluster << " srcNeuronLayoutIndex2: " << srcNeuronLayoutIndex2 << endl;
                    cout << "outIdx: " << outIdx << " outSynIdx: " << outSynIdx << " outCluster: " << outCluster << endl;
                    assert(false);
                }
            }
        }
    }
    )
}

