%	A Cardic arrhythmia classification using stacked ensemble classifier.
%	Authors:
%      Karthikeyan Ramasamy,email-papkarthik@gmail.com
%      Kiruthika Balakrishnan,email-bkiruthikaece@gmail.com
%      Durgadevi Velusamy,email-mvdurgadevi@gmail.com 
close all
warning('off')
N=20; % Population size
lbqr=[1 2 ];% Upper bound
ubqr=[20 20 ];%Lower bound
Dqr=2; %Design variables
MaxIt=200; % Number of iterations
objective = @kfoldloss;  % Objective function
%% Load train and test data
Trainopt=Train;
Trainopt1=Test;
%% Initialization of parameters
for i=1:N
    for j=1:Dqr
    posqr(i,j) =( lbqr(:,j)+rand.*(ubqr(:,j)-lbqr(:,j)));
    end
end
 Q=posqr(:,1);
 r=posqr(:,2);
%% Data
Xtrain=table2array(Trainopt(:,1:end-1)); 
Y_train = Trainopt.N; 
Xtest=table2array(Trainopt1(:,1:end-1)); 
Y_test = Trainopt1.N;
%% Calculation of Objective Function
for i=1:length(posqr(:,1))
for k = 1:length(Xtrain(:,1))     
beta1(i,:) = 2/(Q(i,:)+1);
alpha1(i,:) = 1-beta1(i,:)/r(i,:);
n=length(Xtrain(1,:));
J1(i,:) = floor(log(beta1(i,:)*n/8)/log(1/alpha1(i,:)));
wtrain{k,:}= tqwt_radix2(Xtrain(k,:),Q(i,:),r(i,:),J1(i,:));
ctrain(k,:) = cell2mat(wtrain{k,:});
end
for k = 1:length(Xtest(:,1)) 
wtest{k,:}= tqwt_radix2(Xtest(k,:),Q(i,:),r(i,:),J1(i,:));
ctest(k,:) = cell2mat(wtest{k,:});
end
[coeff,scoretrain,latent,tsquared,explainedtrain,mu] = pca(ctrain);
ctrain=[];
[coeff,scoretest,latent,tsquared,explainedtest,mu] = pca(ctest);
ctest=[];
PCA=find(cumsum(explainedtrain)>96,1);
x_train = scoretrain(:,1:PCA);
x_test=scoretest(:,1:PCA);
lbpca=[2];
ubpca=[PCA];
for p=1:N  
    for q=1
    pospca(p,q) =round( lbpca(:,q)+rand.*(ubpca(:,q)-lbpca(:,q)));
    end
end
x_pca=pospca(:,1);
subspaceDimension = max(1, min(x_pca(i,:), width(x_train) - 1));
 ncol = subspaceDimension;
    x = randperm(size(x_train,2),ncol);
    X_train = x_train(:,x);
    X_test = x_test(:,x);    
%% Prepare learners
linear = templateSVM('KernelFunction', 'linear','PolynomialOrder', [],'KernelScale', 'auto', 'BoxConstraint', 1, 'Standardize', true);
linear_svm = @(x, y)fitcecoc(x, y, 'Learners', linear);
Fine_gaussian = templateSVM('KernelFunction', 'gaussian','PolynomialOrder', [],'KernelScale', 4,'BoxConstraint', 1,'Standardize', true);
Fine_gaussian_svm = @(x, y)fitcecoc(x, y, 'Learners', Fine_gaussian);
Coarse_svm =  templateSVM('KernelFunction', 'gaussian','PolynomialOrder', [],'KernelScale', 63,'BoxConstraint', 1,'Standardize', true);
coarse_svm = @(x, y)fitcecoc(x, y, 'Learners', Coarse_svm);
Medium_gaussian = templateSVM('KernelFunction', 'gaussian','PolynomialOrder', [],'KernelScale', 16,'BoxConstraint', 1,'Standardize', true);
medium_gaussian_svm = @(x, y)fitcecoc(x, y, 'Learners', Medium_gaussian);
Cubic_svm= templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3,'KernelScale', 'auto','BoxConstraint', 1,'Standardize', true);
cubic_svm = @(x, y)fitcecoc(x, y, 'Learners', Cubic_svm);
Quadrtic_svm =  templateSVM('KernelFunction', 'polynomial','PolynomialOrder', 2,'KernelScale', 'auto','BoxConstraint', 1,'Standardize', true);
quadratic_svm = @(x, y)fitcecoc(x, y, 'Learners', Quadrtic_svm);
tree = @(x, y)fitctree(x, y);
knn3 = @(x, y)fitcknn(x, y, 'NumNeighbors', 3);
rf=@(x,y)TreeBagger(100,x,y,'OOBPrediction','On','Method','classification');
learners = {linear_svm, Fine_gaussian_svm, coarse_svm,medium_gaussian_svm, cubic_svm, quadratic_svm};
ens = classification_ensemble(learners); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble

ens_stac = stacking_ensemble(ens, rf); % Initialize Ensemble
ens = ens_stac.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict
acc=100 * sum(y_ens == Y_test) / length(Y_test);
fprintf("Stacking: %.2f%%\n",100 * sum(y_ens == Y_test) / length(Y_test));
f(i,:) = objective(acc);
end

for iter=1:MaxIt    
    
    [fmin, minind] =min(f); %%Xbest
    Xbestqr=posqr(minind,:);
    [fmax, maxind] =max(f);% Xworst
    Xworstqr=posqr(maxind,:); 
    
    [fmin, minind] =min(f); %%Xbest
    Xbestpca=pospca(minind,:);
    [fmax, maxind] =max(f);% Xworst
    Xworstpca=pospca(maxind,:); 
   
for i=1:N             
        Xqr=posqr(i,:);
        Xnewqr= Xqr+rand.*(Xbestqr-abs(Xqr))-rand.*(Xworstqr-abs(Xqr));   
        Xnewqr=max(Xnewqr,lbqr);
        Xnewqr=min(Xnewqr,ubqr);
        
        Xpca=pospca(i,:);
        Xnewpca= round(Xpca+rand.*(Xbestpca-abs(Xpca))-rand.*(Xworstpca-abs(Xpca)));   
        Xnewpca=max(Xnewpca,lbpca);
        Xnewpca=min(Xnewpca,ubpca);

        
Qnew=Xnewqr(:,1);
rnew=Xnewqr(:,2);
xpcanew=round(Xnewpca(:,1));


for k = 1:length(Xtrain(:,1))     
betanew = (2/(Qnew+1));
alphanew = (1-betanew/rnew);
Jnew = floor(log(betanew*n/8)/log(1/alphanew));
wnewtrain{k,:}= tqwt_radix2(Xtrain(k,:),Qnew,rnew,Jnew);
cnewtrain(k,:) = cell2mat(wnewtrain{k,:});
 end

for k = 1:length(Xtest(:,1))
    wnewtest{k,:}= tqwt_radix2(Xtest(k,:),Qnew,rnew,Jnew);
    cnewtest(k,:) = cell2mat(wnewtest{k,:});
end
[coeff,scorenewtrain,latent,tsquared,explainednewtrain,mu] = pca(cnewtrain);
cnewtrain=[];
[coeff,scorenewtest,latent,tsquared,explainednewtest,mu] = pca(cnewtest);
cnewtest=[];
PCAnew=find(cumsum(explainednewtrain)>96,1);
x_train = scorenewtrain(:,1:PCAnew);
x_test=scorenewtest(:,1:PCAnew);

subspaceDimension = max(1, min(xpcanew, width(x_train) - 1));
 ncol = subspaceDimension;
    x = randperm(size(x_train,2),ncol);
    X_train = x_train(:,x);
    X_test = x_test(:,x);
ens = classification_ensemble(learners); % Initialize Ensemble
ens = ens.fit(X_train, Y_train); % Train Ensemble
ens_stac = stacking_ensemble(ens, rf); % Initialize Ensemble
ens = ens_stac.fit(X_train, Y_train); % Train Ensemble
y_ens = ens.predict(X_test); % Predict
acc=100 * sum(y_ens == Y_test) / length(Y_test);
% Print result
fprintf("Stacking: %.2f%%\n",100 * sum(y_ens == Y_test) / length(Y_test));
X_con=[Xnewqr Xnewpca];
pos_con=[posqr pospca];
fnew = objective(acc);


if fnew<f(i)
    pos_con(i,:)=X_con;
    f(i,:)=fnew;
end
end
[optval,optind]=min(f);
bestf(iter)=optval;
best(iter,:)=pos_con(optind,:);
disp(['iteration' num2str(iter) ...
     ':best cost=' num2str(bestf(iter))]); 
 plot(bestf)
 xlabel('The number of iterations');
 ylabel('Fittness function value');
 title('Converegence Plot of Jaya Optimization Algorithm')
end
optvalues = pos_con(optind,:);

 