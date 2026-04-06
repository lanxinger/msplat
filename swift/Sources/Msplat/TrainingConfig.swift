import MsplatCore

/// Configuration for Gaussian splatting training.
public struct TrainingConfig {
    public var iterations: Int32 = 30_000
    public var shDegree: Int32 = 3
    public var shDegreeInterval: Int32 = 1_000
    public var ssimWeight: Float = 0.2
    public var numDownscales: Int32 = 2
    public var resolutionSchedule: Int32 = 3_000
    public var refineEvery: Int32 = 200
    public var warmupLength: Int32 = 0
    public var resetAlphaEvery: Int32 = 30
    public var densifyGradThresh: Float = 0.0002
    public var densifySizeThresh: Float = 0.01
    public var stopScreenSizeAt: Int32 = 4_000
    public var splitScreenSize: Float = 0.05
    public var meanNoiseWeight: Float = 50.0
    public var noiseStopAt: Int32 = 15_000
    public var strategy: Int32 = 0
    public var maxSplats: Int32 = 10_000_000
    public var hybridGrowthFloorDivisor: Float = 0.0
    public var growthGradThreshold: Float = 0.003
    public var growFraction: Float = 0.2
    public var growUntilIter: Int32 = 15_000
    public var opacityDecay: Float = 0.004
    public var scaleDecay: Float = 0.002
    public var boundsPercentile: Float = 0.8
    public var scalesLrInit: Float = 0.007
    public var scalesLrFinal: Float = 0.005
    public var keepCrs: Bool = false
    public var downscaleFactor: Float = 1.0
    /// Background color as (R, G, B) in [0, 1]. Default black, matching Brush-style masked training.
    public var bgColor: (Float, Float, Float) = (0.0, 0.0, 0.0)

    public init() {}

    func toC() -> MsplatConfig {
        var c = msplat_default_config()
        c.iterations = iterations
        c.shDegree = shDegree
        c.shDegreeInterval = shDegreeInterval
        c.ssimWeight = ssimWeight
        c.numDownscales = numDownscales
        c.resolutionSchedule = resolutionSchedule
        c.refineEvery = refineEvery
        c.warmupLength = warmupLength
        c.resetAlphaEvery = resetAlphaEvery
        c.densifyGradThresh = densifyGradThresh
        c.densifySizeThresh = densifySizeThresh
        c.stopScreenSizeAt = stopScreenSizeAt
        c.splitScreenSize = splitScreenSize
        c.meanNoiseWeight = meanNoiseWeight
        c.noiseStopAt = noiseStopAt
        c.strategy = strategy
        c.maxSplats = maxSplats
        c.hybridGrowthFloorDivisor = hybridGrowthFloorDivisor
        c.growthGradThreshold = growthGradThreshold
        c.growFraction = growFraction
        c.growUntilIter = growUntilIter
        c.opacityDecay = opacityDecay
        c.scaleDecay = scaleDecay
        c.boundsPercentile = boundsPercentile
        c.scalesLrInit = scalesLrInit
        c.scalesLrFinal = scalesLrFinal
        c.keepCrs = keepCrs
        c.downscaleFactor = downscaleFactor
        c.bgColor = (bgColor.0, bgColor.1, bgColor.2)
        return c
    }
}
