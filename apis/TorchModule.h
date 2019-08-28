//
//  TorchModule.h
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class TorchIValue;
@interface TorchModule : NSObject

+ (TorchModule* _Nullable)loadTorchscriptModel:(NSString* _Nullable)modelPath;

- (TorchIValue* _Nullable)forward:(NSArray<TorchIValue* >* _Nullable)values;

- (TorchIValue* _Nullable)run_method:(NSString* _Nullable)methodName withInputs:(NSArray<TorchIValue* >* _Nullable) inputs;

@end

NS_ASSUME_NONNULL_END
