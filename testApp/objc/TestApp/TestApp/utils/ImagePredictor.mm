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
    PTHTensor* imageTensor = [PTHTensor newWithType:PTHTensorTypeByte Size:@[ @(1), @(IMG_W), @(IMG_H), @(IMG_C) ] Data:(void* )pixels.bytes];
    imageTensor = [imageTensor permute:@[@(0),@(3),@(1),@(2)]];
    imageTensor = [imageTensor to:PTHTensorTypeFloat];
    //normalize the tensor
    imageTensor = [imageTensor div_:255.0];
    [[imageTensor[0][0] sub_:0.485] div_:0.229];
    [[imageTensor[0][1] sub_:0.485] div_:0.229];
    [[imageTensor[0][2] sub_:0.485] div_:0.229];
    PTHIValue* inputIValue = [PTHIValue newIValueWithTensor:imageTensor];
    PTHTensor* outputTensor = [[_module forward:@[inputIValue]] toTensor];
    //collect the top10 results
    NSArray<PTHTensor* >* topkResults = [outputTensor topKResult:@(10) Dim:@(-1) isLargest:YES isSorted:YES];
    PTHTensor* scores = [topkResults[0] view:@[@(-1)]];
    PTHTensor* idxs   = [topkResults[1] view:@[@(-1)]];
    NSMutableArray* topScores = [NSMutableArray new];
    NSMutableArray* topIndexes = [NSMutableArray new];
    for(int i=0;i<10;++i){
        [topScores addObject:@(scores[i].item.floatValue)];
        [topIndexes addObject:@(idxs[i].item.longValue)];
    }
    if(completion){
        completion(topScores.copy, topIndexes.copy);
    }
}


@end
