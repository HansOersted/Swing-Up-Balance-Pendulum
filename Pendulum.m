clear
clc
close all

%% 

env = rlPredefinedEnv("SimplePendulumWithImage-Continuous");

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);


%% Create DDPG Agent

hiddenLayerSize1 = 256;
hiddenLayerSize2 = 256;

% Image input path
imgPath = [
    imageInputLayer(obsInfo(1).Dimension, ...
        Name="imgInLyr")
    convolution2dLayer(5,8,Stride=3,Padding=0) % (16x16x8)
    reluLayer
    convolution2dLayer(5,8,Stride=3,Padding=0) % (4x4x8)
    reluLayer    
    fullyConnectedLayer(32)
    concatenationLayer(1,2,Name="cat1")
    fullyConnectedLayer(hiddenLayerSize1)
    reluLayer
    fullyConnectedLayer(hiddenLayerSize2)
    additionLayer(2,Name="add")
    reluLayer
    fullyConnectedLayer(1,Name="fc4")
    ];

% d(theta)/dt input path
dthPath = [
    featureInputLayer(prod(obsInfo(2).Dimension), ...
        Name="dthInLyr")
    fullyConnectedLayer(1,Name="fc5", ...
        BiasLearnRateFactor=0, ...
        Bias=0)
    ];

% Action path
actPath =[
    featureInputLayer(prod(obsInfo(2).Dimension), ...
        Name="actInLyr")
    fullyConnectedLayer(hiddenLayerSize2, ...
        Name="fc6", ...
        BiasLearnRateFactor=0, ...
        Bias=zeros(hiddenLayerSize2,1))
    ];

criticNetwork = dlnetwork();
criticNetwork = addLayers(criticNetwork,imgPath);
criticNetwork = addLayers(criticNetwork,dthPath);
criticNetwork = addLayers(criticNetwork,actPath);
criticNetwork = connectLayers(criticNetwork,"fc5","cat1/in2");
criticNetwork = connectLayers(criticNetwork,"fc6","add/in2");

plot(criticNetwork)

critic = rlQValueFunction(criticNetwork, ...
    obsInfo,actInfo,...
    ObservationInputNames=["imgInLyr","dthInLyr"], ...
    ActionInputNames="actInLyr");

%%

% Image input path
imgPath = [
    imageInputLayer(obsInfo(1).Dimension, ...        
        Name="imgInLyr")
    convolution2dLayer(5,8,Stride=3,Padding=0)
    reluLayer
    convolution2dLayer(5,8,Stride=3,Padding=0)
    reluLayer    
    fullyConnectedLayer(32)
    concatenationLayer(1,2,Name="cat1")
    fullyConnectedLayer(hiddenLayerSize1)
    reluLayer
    fullyConnectedLayer(hiddenLayerSize2)
    reluLayer
    fullyConnectedLayer(1)
    tanhLayer
    scalingLayer(Name="scale1", ...
        Scale=max(actInfo.UpperLimit))
    ];

% d(theta)/dt input layer
dthPath = [
    featureInputLayer(prod(obsInfo(2).Dimension), ...
        Name="dthInLyr")
    fullyConnectedLayer(1, ...
        Name="fc5", ...
        BiasLearnRateFactor=0, ...
        Bias=0) 
    ];


actorNetwork = dlnetwork();
actorNetwork = addLayers(actorNetwork,imgPath);
actorNetwork = addLayers(actorNetwork,dthPath);
actorNetwork = connectLayers(actorNetwork,"fc5","cat1/in2");


figure
plot(actorNetwork)


actorNetwork = initialize(actorNetwork);
summary(actorNetwork)


actor = rlContinuousDeterministicActor(actorNetwork, ...
    obsInfo,actInfo, ...
    ObservationInputNames=["imgInLyr","dthInLyr"]);


%%

criticOptions = rlOptimizerOptions( ...
    LearnRate=1e-03, ...
    GradientThreshold=1);
actorOptions = rlOptimizerOptions( ...
    LearnRate=1e-04, ...
    GradientThreshold=1);


UseGPUCritic = false;
if canUseGPU && UseGPUCritic    
    critic.UseDevice = "gpu";
end

UseGPUActor = false;
if canUseGPU && UseGPUActor    
    actor.UseDevice = "gpu";
end


if canUseGPU && (UseGPUCritic || UseGPUActor)
    gpurng(0)
end


agentOptions = rlDDPGAgentOptions(...
    SampleTime=env.Ts,...
    TargetSmoothFactor=1e-3,...
    ExperienceBufferLength=1e6,...
    DiscountFactor=0.99,...
    MiniBatchSize=128);


agentOptions.NoiseOptions.StandardDeviation = 0.6;
agentOptions.NoiseOptions.StandardDeviationDecayRate = 1e-6;
agentOptions.NoiseOptions.StandardDeviationMin = 0.1;


agentOptions.CriticOptimizerOptions = criticOptions;
agentOptions.ActorOptimizerOptions = actorOptions;

agent = rlDDPGAgent(actor,critic,agentOptions);


%% Train Agent

maxepisodes = 5000;
maxsteps = 400;
trainingOptions = rlTrainingOptions(...
    MaxEpisodes=maxepisodes,...
    MaxStepsPerEpisode=maxsteps,...
    Plots="training-progress",...
    StopTrainingCriteria="EvaluationStatistic",...
    StopTrainingValue=-740); % reward has been determined in the env

evl = rlEvaluator(EvaluationFrequency=50, NumEpisodes=1);

plot(env)


doTraining = false;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions, Evaluator=evl);
else
    % Load pretrained agent for the example.
    load("SimplePendulumWithImageDDPG.mat","agent")       
end


%% Simulate DDPG Agent

rng(1); % For reproducibility
simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);


