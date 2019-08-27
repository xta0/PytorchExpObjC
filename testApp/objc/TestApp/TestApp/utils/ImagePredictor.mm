//
//  Copyright © 2019年 Vizlab. All rights reserved.
//

#import "ImagePredictor.h"
#import <torch/script.h>
#import <PytorchExpObjC/PytorchExpObjC.h>
//#import <PytorchExpObjC/PTHIValue+Internal.h>
//#import <PytorchExpObjC/PTHTensor+Internal.h>
#import <vector>
#import "UIImage+Utils.h"

#define IMG_W 224
#define IMG_H 224
#define IMG_C 3

@implementation ImagePredictor {
    PTHModule* _module;
    NSArray* _labels;
}

- (instancetype)initWithModelPath:(NSString* )modelPath {
    self = [super init];
    if (self) {
        _module = [PTHModule loadTorchscriptModel:modelPath];
        NSError* err;
        NSString* str = [NSString stringWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        _labels = [str componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    }
    return self;
}
- (void)predict:(UIImage* )image
     Completion:(void(^__nullable)(NSArray<NSNumber* >* scores,
                                   NSArray<NSNumber*>* index)) completion {
    
    NSData* pixels = [image resize:{IMG_W,IMG_H}].rgb;
    PTHTensor* imageTensor = [PTHTensor newWithType:PTHTensorTypeByte Shape:@[ @(1), @(IMG_W), @(IMG_H), @(IMG_C) ] Data:(void* )pixels.bytes];
    imageTensor = [imageTensor permute:@[@(0),@(3),@(1),@(2)]];
    imageTensor = [imageTensor toType:PTHTensorTypeFloat];
    
    //---------------------------------------------------------------------
    //calling private APIs, cuz I didn't implement those ops below.
    auto img_tensor = [imageTensor toTensor];
    img_tensor.div_(255);
    // normalize the input tensor
    img_tensor[0][0].sub_(0.485).div_(0.229);
    img_tensor[0][1].sub_(0.456).div_(0.224);
    img_tensor[0][2].sub_(0.406).div_(0.225);
    //---------------------------------------------------------------------
    
    imageTensor = [PTHTensor newWithTensor:img_tensor];
    PTHIValue* inputIValue = [PTHIValue newIValueWithTensor:imageTensor];
    PTHIValue* outputIValue = [_module forward:@[inputIValue]];
    PTHTensor* outputTensor = [outputIValue toTensor];
    
    //---------------------------------------------------------------------
    //calling private APIs, cuz I didn't implement those ops below.
    auto outputs = [outputTensor toTensor];
    auto result = outputs.topk(10, -1);
    //flat socres and indexes
    auto scores = std::get<0>(result).view(-1);
    auto idxs = std::get<1>(result).view(-1);
    NSMutableArray* topScores = [NSMutableArray new];
    NSMutableArray* topIndexes = [NSMutableArray new];
    //collect top 10 results
    for (int i = 0; i < 10; ++i) {
        [topScores addObject:@(scores[i].item().toFloat())];
        [topIndexes addObject:@(idxs[i].item().toInt())];
    }
    //---------------------------------------------------------------------
    if(completion){
        completion(topScores.copy, topIndexes.copy);
    }
}


@end
