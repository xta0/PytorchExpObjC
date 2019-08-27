//
//  PTHModule.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class PTHIValue;
@interface PTHModule : NSObject

+ (PTHModule* _Nullable)loadTorchscriptModel:(NSString* _Nullable)modelPath;

- (PTHIValue* _Nullable)forward:(NSArray<PTHIValue* >* _Nullable)values;

- (PTHIValue* _Nullable)run_method:(NSString* _Nullable)methodName withInputs:(NSArray<PTHIValue* >* _Nullable) inputs;

@end

NS_ASSUME_NONNULL_END
