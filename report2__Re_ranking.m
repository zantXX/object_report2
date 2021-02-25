function [sorted_score,sorted_idx] = report2__Re_ranking(training_posDB,training_negDB,val_DB)
    train_imgs = [];
    val_imgs = [];
    net = alexnet;
    %学習画像読み込み
    for j=1:size(training_posDB,2)
        img = imread(training_posDB{j});
        reimg = imresize(img,net.Layers(1).InputSize(1:2));
        train_imgs=cat(4,train_imgs,reimg);
    end
    for j=1:size(training_negDB,2)
        img = imread(training_negDB{j});
        reimg = imresize(img,net.Layers(1).InputSize(1:2));
        train_imgs=cat(4,train_imgs,reimg);
    end
    
    %評価画像読み込み
    for j=1:size(val_DB,2)
        img = imread(val_DB{j});
        reimg = imresize(img,net.Layers(1).InputSize(1:2));
        val_imgs=cat(4,val_imgs,reimg);
    end
    
    %ラベル作成
    training_label=[ones(size(training_posDB,2),1); ones(size(training_negDB,2),1)*(-1)];
    
    %DNN準備
    train_dcnnf = activations(net,train_imgs,'fc7');  
    train_dcnnf = squeeze(train_dcnnf);
    train_dcnnf = train_dcnnf/norm(train_dcnnf);
    train_dcnnf =train_dcnnf';
    
    val_dcnnf = activations(net,val_imgs,'fc7');  
    val_dcnnf = squeeze(val_dcnnf);
    val_dcnnf = val_dcnnf/norm(val_dcnnf);
    val_dcnnf =val_dcnnf';
    
    %SVM(linear)
    model = fitcsvm(train_dcnnf, training_label,'KernelFunction','linear','KernelScale','auto');
    [~,score]=predict(model,val_dcnnf);
    %ソート
    [sorted_score,sorted_idx] = sort(score(:,2),'descend');
    %出力
    for i=1:numel(sorted_idx)
        fprintf('%s %f\n',val_DB{sorted_idx(i)},sorted_score(i));
    end
end