package edu.uwb.braingrid.workbench.model;

/**
 * Maintains data for an input configuration.
 * @author Del Davis
 */
public class InputConfiguration {

    /* Config State Data */
    private String lsmFracExc;
    private String lsmStartNeurons;
    private String poolSizeX;
    private String poolSizeY;
    private String poolSizeZ;
    private String iInjectMin;
    private String iInjectMax;
    private String iNoiseMin;
    private String iNoiseMax;
    private String vThreshMin;
    private String vThreshMax;
    private String vRestingMin;
    private String vRestingMax;
    private String vResetMin;
    private String vResetMax;
    private String vInitMin;
    private String vInitMax;
    private String starterVThreshMin;
    private String starterVThreshMax;
    private String starterVResetMin;
    private String starterVResetMax;
    private String growthParamsEpsilon;
    private String growthBeta;
    private String growthParamsRho;
    private String growthParamsTargetRate;
    private String growthParamsMinRadius;
    private String growthParamsStartRadius;
    private String simParamsTSim;
    private String simParamsNumSims;
    private String simParamsMaxFiringRate;
    private String simParamsMaxSynapsesPerNeuron;
    private String outputParamsStateOutputFileName;
    private String seedValue;
    private String layoutType;
    private String layoutFilesActiveNListFileName;
    private String layoutFilesInhNListFileName;

    public InputConfiguration() {
        lsmFracExc = null;
        lsmStartNeurons = null;
        poolSizeX = null;
        poolSizeY = null;
        poolSizeZ = null;
        iInjectMin = null;
        iInjectMax = null;
        iNoiseMin = null;
        iNoiseMax = null;
        vThreshMin = null;
        vThreshMax = null;
        vRestingMin = null;
        vRestingMax = null;
        vResetMin = null;
        vResetMax = null;
        vInitMin = null;
        vInitMax = null;
        starterVThreshMin = null;
        starterVThreshMax = null;
        starterVResetMin = null;
        starterVResetMax = null;
        growthParamsEpsilon = null;
        growthBeta = null;
        growthParamsRho = null;
        growthParamsTargetRate = null;
        growthParamsMinRadius = null;
        growthParamsStartRadius = null;
        simParamsTSim = null;
        simParamsNumSims = null;
        simParamsMaxFiringRate = null;
        simParamsMaxSynapsesPerNeuron = null;
        outputParamsStateOutputFileName = null;
        seedValue = null;
        layoutType = null;
        layoutFilesActiveNListFileName = null;
        layoutFilesInhNListFileName = null;
    }

    private void setLSMFracExc(String value) {
        lsmFracExc = value;
    }

    private String getLSMFracExc() {
        return lsmFracExc;
    }

    private void setLSMStartNeurons(String value) {
        lsmStartNeurons = value;
    }

    private String getLSMStartNeurons() {
        return lsmStartNeurons;
    }

    private void setPoolSizeX(String value) {
        poolSizeX = value;
    }

    private String getPoolSizeX() {
        return poolSizeX;
    }

    private void setPoolSizeY(String value) {
        poolSizeY = value;
    }

    private String getPoolSizeY() {
        return poolSizeY;
    }

    private void setPoolSizeZ(String value) {
        poolSizeZ = value;
    }

    private String getPoolSizeZ() {
        return poolSizeZ;
    }

    private void setIInjectMin(String value) {
        iInjectMin = value;
    }

    private String getIInjectMin() {
        return iInjectMin;
    }

    private void setIInjectMax(String value) {
        iInjectMax = value;
    }

    private String getIInjectMax() {
        return iInjectMax;
    }

    private void setINoiseMin(String value) {
        iNoiseMin = value;
    }

    private String getINoiseMin() {
        return iNoiseMin;
    }

    private void setINoiseMax(String value) {
        iNoiseMax = value;
    }

    private String getINoiseMax() {
        return iNoiseMax;
    }

    private void setVThreshMin(String value) {
        vThreshMin = value;
    }

    private String getVThreshMin() {
        return vThreshMin;
    }

    private void setVThreshMax(String value) {
        vThreshMax = value;
    }

    private String getVThreshMax() {
        return vThreshMax;
    }

    private void setVRestingMin(String value) {
        vRestingMin = value;
    }

    private String getVRestingMin() {
        return vRestingMin;
    }

    private void setVRestingMax(String value) {
        vRestingMax = value;
    }

    private String getVRestingMax() {
        return vRestingMax;
    }

    private void setVResetMin(String value) {
        vResetMin = value;
    }

    private String getVResetMin() {
        return vResetMin;
    }

    private void setVResetMax(String value) {
        vResetMax = value;
    }

    private String getVResetMax() {
        return vResetMax;
    }

    private void setVInitMin(String value) {
        vInitMin = value;
    }

    private String getVInitMin() {
        return vInitMin;
    }

    private void setVInitMax(String value) {
        vInitMax = value;
    }

    private String getVInitMax() {
        return vInitMax;
    }

    private void setStarterVThreshMin(String value) {
        starterVThreshMin = value;
    }

    private String getStarterVThreshMin() {
        return starterVThreshMin;
    }

    private void setStarterVThreshMax(String value) {
        starterVThreshMax = value;
    }

    private String getStarterVThreshMax() {
        return starterVThreshMax;
    }

    private void setStarterVResetMin(String value) {
        starterVResetMin = value;
    }

    private String getStarterVResetMin() {
        return starterVResetMin;
    }

    private void setStarterVResetMax(String value) {
        starterVResetMax = value;
    }

    private String getStarterVResetMax() {
        return starterVResetMax;
    }

    private void setGrowthParamsEpsilon(String value) {
        growthParamsEpsilon = value;
    }

    private String getGrowthParamsEpsilon() {
        return growthParamsEpsilon;
    }

    private void setGrowthBeta(String value) {
        growthBeta = value;
    }

    private String getGrowthBeta() {
        return growthBeta;
    }

    private void setGrowthParamsRho(String value) {
        growthParamsRho = value;
    }

    private String getGrowthParamsRho() {
        return growthParamsRho;
    }

    private void setGrowthParamsTargetRate(String value) {
        growthParamsTargetRate = value;
    }

    private String getGrowthParamsTargetRate() {
        return growthParamsTargetRate;
    }

    private void setGrowthParamsMinRadius(String value) {
        growthParamsMinRadius = value;
    }

    private String getGrowthParamsMinRadius() {
        return growthParamsMinRadius;
    }

    private void setGrowthParamsStartRadius(String value) {
        growthParamsStartRadius = value;
    }

    private String getGrowthParamsStartRadius() {
        return growthParamsStartRadius;
    }

    private void setSimParamsTSim(String value) {
        simParamsTSim = value;
    }

    private String getSimParamsTSim() {
        return simParamsTSim;
    }

    private void setSimParamsNumSims(String value) {
        simParamsNumSims = value;
    }

    private String getSimParamsNumSims() {
        return simParamsNumSims;
    }

    private void setSimParamsMaxFiringRate(String value) {
        simParamsMaxFiringRate = value;
    }

    private String getSimParamsMaxFiringRate() {
        return simParamsMaxFiringRate;
    }

    private void setSimParamsMaxSynapsesPerNeuron(String value) {
        simParamsMaxSynapsesPerNeuron = value;
    }

    private String getSimParamsMaxSynapsesPerNeuron() {
        return simParamsMaxSynapsesPerNeuron;
    }

    private void setOutputParamsStateOutputFileName(String value) {
        outputParamsStateOutputFileName = value;
    }

    private String getOutputParamsStateOutputFileName() {
        return outputParamsStateOutputFileName;
    }

    private void setSeedValue(String value) {
        seedValue = value;
    }

    private String getSeedValue() {
        return seedValue;
    }

    private void setLayoutType(String value) {
        layoutType = value;
    }

    private String getLayoutType() {
        return layoutType;
    }

    private void setLayoutFilesActiveNListFileName(String value) {
        layoutFilesActiveNListFileName = value;
    }

    private String getLayoutFilesActiveNListFileName() {
        return layoutFilesActiveNListFileName;
    }

    private void setLayoutFilesInhNListFileName(String value) {
        layoutFilesInhNListFileName = value;
    }

    private String getLayoutFilesInhNListFileName() {
        return layoutFilesInhNListFileName;
    }
}
