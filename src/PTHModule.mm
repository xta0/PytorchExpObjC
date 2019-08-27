//
//  PTHModule.m
//  Pytorch-Exp-Demo
//
//  Created by taox on 8/25/19.
//  Copyright © 2019 taox. All rights reserved.
//

#import <PytorchExp/PytorchExp.h>
#import "PTHModule.h"
#import "PTHIValue+Internal.h"


@implementation PTHModule {
    std::shared_ptr<torch::jit::script::Module> _impl;
}

+ (PTHModule* _Nullable)loadTorchscriptModel:(NSString* _Nullable)modelPath {
    if(modelPath.length == 0){
        return nil;
    }
    auto torchScriptModule = torch::jit::load([modelPath cStringUsingEncoding:NSASCIIStringEncoding]);
    auto impl = std::make_shared<torch::jit::script::Module>(torchScriptModule);
    if (!impl) {
        return nil;
    }
    PTHModule* module = [PTHModule new];
    module->_impl = std::move(impl);
    return module;
}

- (PTHIValue* _Nullable)forward:(NSArray<PTHIValue* >* _Nullable)values {
    if (values.count == 0){
        return nil;
    }
    std::vector<at::IValue> inputs;
    for(PTHIValue* value in values) {
        inputs.push_back(value.toIValue);
    }
    auto result = _impl->forward(inputs);
    return [PTHIValue newWithIValue:result];
}


- (PTHIValue* _Nullable)run_method:(NSString* _Nullable)methodName withInputs:(NSArray<PTHIValue* >* _Nullable) values {
    if (methodName.length == 0 || values.count ==0 ) {
        return nil;
    }
    std::vector<at::IValue> inputs;
    for(PTHIValue* value in values) {
        inputs.push_back(value.toIValue);
    }
    if (auto method = _impl->find_method(std::string([methodName cStringUsingEncoding:NSASCIIStringEncoding]))){
        auto result = (*method)(std::move(inputs));
        return [PTHIValue newWithIValue:result];
    }
    // raise an exception?
    return nil;
}

- (void)dealloc {
}


@end
