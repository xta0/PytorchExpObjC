//
//  ViewController.m
//  TestApp
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import "ViewController.h"
#import "ImagePredictor.h"

@interface ViewController (){
    ImagePredictor* _predictor;
    NSArray* _labels;
}

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    _predictor = [[ImagePredictor alloc]initWithModelPath:[[NSBundle mainBundle] pathForResource:@"resnet18" ofType:@"pt"]];
    _labels    = [[NSString stringWithContentsOfFile:[[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"]
                                          encoding:NSUTF8StringEncoding
                                             error:nil] componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];

    
    UIImage* image = [UIImage imageNamed:@"wolf_400x400.jpg"];
    __weak ViewController* weakSelf = self;
    [_predictor predict:image Completion:^(NSArray<NSNumber *> * _Nonnull scores, NSArray<NSNumber *> * _Nonnull indexes) {
        __strong ViewController* strongSelf = weakSelf;
        if(strongSelf) {
            if(scores.count > 0 && indexes.count > 0 && strongSelf->_labels.count > 0){
                NSString* result = @"";
                for(int i=0; i<indexes.count; ++i){
                    NSString* label = strongSelf->_labels[indexes[i].integerValue];
                    NSString* content = [NSString stringWithFormat:@"- [Score]: %@, [label]: %@ \n\n", scores[i], label];
                    result = [result stringByAppendingString:content];
                }
                dispatch_async(dispatch_get_main_queue(), ^{
                    NSLog(@"%@",result);
                });
            }
        }
    }];
}


@end
