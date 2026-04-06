#ifndef MODEL_H
#define MODEL_H

#include "metal_tensor.hpp"
#include "ssim.hpp"
#include "input_data.hpp"

int numShBases(int degree);
float psnr(const MTensor& rendered, const MTensor& gt, const MTensor* mask = nullptr);
float l1_loss(const MTensor& rendered, const MTensor& gt, const MTensor* mask = nullptr);

enum class Strategy {
  Classic = 0,
  Hybrid = 1,
  MRNF = 2,
  IGSPlus = 3,
};

struct Model{
  Model(const InputData &inputData, int numCameras,
        int numDownscales, int resolutionSchedule, int shDegree, int shDegreeInterval,
        int refineEvery, int warmupLength, int resetAlphaEvery, float densifyGradThresh, float densifySizeThresh, int stopScreenSizeAt, float splitScreenSize,
        int maxSteps, bool keepCrs, float meanNoiseWeight, int noiseStopAt,
        bool mipSplatting = false,
        Strategy strategy = Strategy::Classic, int maxSplats = 1000000, float hybridGrowthFloorDivisor = 0.0f,
        float growthGradThreshold = 0.003f, float growFraction = 0.2f, int growUntilIter = 15000,
        float opacityDecay = 0.004f, float scaleDecay = 0.002f, float boundsPercentile = 0.8f,
        float scalesLrInit = 0.007f, float scalesLrFinal = 0.005f,
        const float* bgColor = nullptr);

  ~Model(){ releaseOptimizers(); }

  void setupOptimizers();
  void releaseOptimizers();

  void schedulersStep(int step);
  int getDownscaleFactor(int step);
  void afterTrain(std::vector<Camera>& cameras, int step);
  void applyMaskOpacityPenalty(std::vector<Camera>& cameras, int step);
  void save(const std::string &filename, int step);
  void savePly(const std::string &filename, int step);
  void saveSplat(const std::string &filename);
  int loadPly(const std::string &filename);
  void saveCheckpoint(const std::string &filename, int step);
  int loadCheckpoint(const std::string &filename);
  struct CamSetup {
    float fx, fy, cx, cy;
    int height, width, degree, degreesToUse;
    std::tuple<int,int,int> tileBounds;
    float cam_pos[3];
  };
  CamSetup prepareCam(Camera& cam, int step);
  void fullIteration(Camera& cam, int step, MTensor &gt, float ssimWeight, MTensor *mask = nullptr);
  MTensor render(Camera& cam, int step);
  void applyMeanNoise(int step);
  void computeBounds();
  void computeBudgetSchedule();

  struct SceneBounds {
    float center[3] = {};
    float extent[3] = {};
    float medianSize = 0.0f;
    float maxExtent = 0.0f;
  };

  MTensor means;
  MTensor scales;
  MTensor quats;
  MTensor featuresDc;
  MTensor featuresRest;
  MTensor opacities;

  static constexpr int N_ADAM_GROUPS = 6;
  MTensor adam_exp_avg[N_ADAM_GROUPS];
  MTensor adam_exp_avg_sq[N_ADAM_GROUPS];
  int adam_step_count = 0;
  float adam_lr[N_ADAM_GROUPS] = {};
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
  float means_lr_init = 0, means_lr_final = 0;

  MTensor means_buf, scales_buf, quats_buf, featuresDc_buf, featuresRest_buf, opacities_buf;
  MTensor adam_exp_avg_buf[N_ADAM_GROUPS], adam_exp_avg_sq_buf[N_ADAM_GROUPS];
  int num_active = 0, buf_capacity = 0;
  void refreshViews();
  void ensureCapacity(int needed);

  MTensor densify_split_flag, densify_dup_flag;
  MTensor densify_split_prefix, densify_dup_prefix;
  MTensor densify_keep_flag, densify_keep_prefix;
  MTensor densify_keep_count_readback;
  MTensor densify_block_totals;
  MTensor densify_compact_scratch;
  MTensor donorIndexScratch;
  std::vector<uint8_t> donorChosenScratch;
  std::vector<double> donorWeightScratchA;
  std::vector<double> donorWeightScratchB;

  MTensor radii;
  int lastHeight;
  int lastWidth;

  MTensor xysGradNorm;
  MTensor visCounts;
  MTensor max2DSize;
  MTensor refineWeightMax;
  MTensor errorScoreMax;

  MTensor backgroundColor;
  MTensor window2d;  // SSIM window (11,11) f32

  int numCameras;
  int numDownscales;
  int resolutionSchedule;
  int shDegree;
  int shDegreeInterval;
  int refineEvery;
  int warmupLength;
  int resetAlphaEvery;
  int stopSplitAt;
  float densifyGradThresh;
  float densifySizeThresh;
  int stopScreenSizeAt;
  float splitScreenSize;
  int maxSteps;
  bool keepCrs;
  bool mipSplatting;
  float meanNoiseWeight;
  int noiseStopAt;
  Strategy strategy;
  bool hybridRefine;
  int maxSplats;
  float hybridGrowthFloorDivisor;
  float growthGradThreshold;
  float growFraction;
  int growUntilIter;
  float opacityDecay;
  float scaleDecay;
  float boundsPercentile;
  SceneBounds bounds;
  bool boundsValid = false;
  int refineWindowsSinceBounds = 0;
  float meansLrUnscaled = 0.0f;
  float scaleLrCurrent = 0.0f;
  double meansLrGamma = 1.0;
  double scaleLrGamma = 1.0;
  std::vector<int64_t> budgetSchedule;
  int igsCurrentStep = 0;
  int igsTotalSteps = 0;
  int64_t igsInitialPoints = 0;
  bool maskAwareData = false;

  float scale;
  float translation[3] = {};
};

#endif
