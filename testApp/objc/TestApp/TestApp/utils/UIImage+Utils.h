//
//  Copyright © 2019年 Vizlab. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <memory>

NS_ASSUME_NONNULL_BEGIN

@interface UIImage (Utils)

- (NSData* )rgb;
- (UIImage* )resize:(CGSize)sz;
- (UIImage* )rgbImage;


@end

NS_ASSUME_NONNULL_END
