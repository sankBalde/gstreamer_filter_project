set(CMAKE_CUDA_COMPILER "/run/current-system/sw/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "/nix/store/sw9kw1fz9h7h0pr2hk7jyx8m1ac1ydsg-gcc-wrapper-11.3.0/bin/g++")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/nix/store/sw9kw1fz9h7h0pr2hk7jyx8m1ac1ydsg-gcc-wrapper-11.3.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.7.64")
set(CMAKE_CUDA_DEVICE_LINKER "/run/current-system/sw/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/run/current-system/sw/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.3")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/run/current-system/sw")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/run/current-system/sw")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.7.64")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/run/current-system/sw")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "75-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/run/current-system/sw/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/run/current-system/sw/targets/x86_64-linux/lib/stubs;/run/current-system/sw/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/run/current-system/sw/include;/nix/store/1k2kggmhb49355snwqcwdgq1wlxdh3iz-curl-8.1.1-dev/include;/nix/store/mlvv3yjci2vjf3b1nqf03zjh13haclk0-brotli-1.0.9-dev/include;/nix/store/iy9i2vrbmsq91i66wwx1ijn4d2j3wmy1-libkrb5-1.20.1-dev/include;/nix/store/78fwcm3kwyahbindmk1ksygcwjlxx1gn-nghttp2-1.51.0-dev/include;/nix/store/dkdmab6s9svzs9k56pggc45q2h0qmcp7-libidn2-2.3.4-dev/include;/nix/store/sss6wmnvh82nyhf9024bdxgqi6k0ygm2-openssl-3.0.10-dev/include;/nix/store/hjykkd5b3ab3690hpan4018xmcqcj3x9-libssh2-1.11.0-dev/include;/nix/store/03grn1z9pl9m8702wkshj5pwnvknazim-zlib-1.2.13-dev/include;/nix/store/nj5b4bppgf75yk8jzdb4ckk4asjk9b7z-zstd-1.5.5-dev/include;/nix/store/ald87b4irf657al0idll1vfggg0m1p6k-procps-3.3.17/include;/nix/store/5dh274aahjdbcmgx6plx6jjf0qadhg28-gnumake-4.4.1/include;/nix/store/izxd08ayvffqdd5xgp54jkx7p4vpmfhq-util-linux-2.38.1-dev/include;/nix/store/s8pmhzg5jnmhgm03pjdyj9bpgnz9s8kv-glu-9.0.2-dev/include;/nix/store/chfsxwdy3ai6b3g8pcr95rsfw41n15zj-libGL-1.6.0-dev/include;/nix/store/qrs00hnv9frpk49rbp6dir3vqw0my8j7-libXi-1.8-dev/include;/nix/store/mkkj3rwhy4xhwbzdpfjhzci0sp19vbsq-libXfixes-6.0.0-dev/include;/nix/store/ig0cmlw59834hpmk0v6azipl0ss7lxcb-libXext-1.3.4-dev/include;/nix/store/245c0wsqvmkz7wrcbcw2zwcjvgd42v4b-xorgproto-2021.5/include;/nix/store/q0fh1fgyyj43x8nfjn2fyxwr6xki9pf0-libXau-1.0.9-dev/include;/nix/store/401i1waiyn8w40r6058q3p517cwgb1vc-libXmu-1.1.3-dev/include;/nix/store/kp13ab0k35fq4ffbkygip2mxm30lfwjb-libX11-1.8.6-dev/include;/nix/store/wmhp81hly5bghkayfhp7r692qq4qnjnc-libxcb-1.14-dev/include;/nix/store/rk0a0frvq0kfqaal0z6nv3g4lwcwsjm3-libXt-1.2.1-dev/include;/nix/store/igq9pr2bzrrdnw6faybzagx37gzypzyr-libSM-1.2.3-dev/include;/nix/store/3hivwkkd8wx2sa3kb9br5k39w2gb5wy2-libICE-1.0.10-dev/include;/nix/store/xvz26ns18v31fs2kibahhgyrx4p9rih4-freeglut-3.4.0-dev/include;/nix/store/2g7jmjjwlqhgyan75j53rbwmfb1mk949-libXv-1.0.11-dev/include;/nix/store/ai9nkb894rrmdpargwz4g1144ka81zln-libXrandr-1.5.2-dev/include;/nix/store/afzk1lvv5306hnsbarq1ww9j655060vq-libXrender-0.9.10-dev/include;/nix/store/7l5i6rd8a19mkx7h593ckm6q9fa2k98z-pngpp-0.2.10/include;/nix/store/r9isndk4r4vrrcg45zawd5prlgswxbgg-libpng-apng-1.6.39-dev/include;/nix/store/nfrf5bm2p5rz0k4pyaaffx5vv0b961d9-tbb-2020.3-dev/include;/nix/store/19s69x56fidcyscaqm8sdf87hswaz8rg-gbenchmark-1.7.1/include;/nix/store/kd9r4vwxvgbpyrqaz76s1dddxybj339h-ncurses-abi5-compat-6.4-dev/include;/nix/store/j1sd8bp4gha25mn5pmjldycf1fwzbl3p-gstreamer-1.22.5-dev/include;/nix/store/dypzmzgjvyfs32gg8sj0rb6q79nf4390-glib-2.76.2-dev/include;/nix/store/798wjzmkdxczl9jlvbsi95f0vfhd33fq-libffi-3.4.4-dev/include;/nix/store/fgr89gfh5if2x12qm0wz2s996jyn5qx9-gettext-0.21/include;/nix/store/wd2678n4jhz1clp28d2niyliv4smjhbm-glibc-iconv-2.37/include;/nix/store/90mfdml0m28gkw68h7i033v667v82ghm-gst-plugins-base-1.22.5-dev/include;/nix/store/6clbdpf67kwg8riagqf88cdzsbiaxsdy-gst-plugins-bad-1.22.5-dev/include;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/include/c++/11.3.0;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/include/c++/11.3.0/x86_64-unknown-linux-gnu;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/include/c++/11.3.0/backward;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/lib/gcc/x86_64-unknown-linux-gnu/11.3.0/include;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/include;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/lib/gcc/x86_64-unknown-linux-gnu/11.3.0/include-fixed;/nix/store/hkj250rjsvxcbr31fr1v81cv88cdfp4l-glibc-2.37-8-dev/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/run/current-system/sw/targets/x86_64-linux/lib/stubs;/run/current-system/sw/targets/x86_64-linux/lib;/run/current-system/sw/lib;/nix/store/2av8x1ipwfm5n8nrx1i18bygza6v8b6g-brotli-1.0.9-lib/lib;/nix/store/wlacmv33nsrf2vnfa2akcym13zh6zzrx-libkrb5-1.20.1/lib;/nix/store/7aw32rkwgbim37x6kgza7qvr35czjysa-nghttp2-1.51.0-lib/lib;/nix/store/4563gldw8ibz76f1a3x69zq3a1vhdpz9-libidn2-2.3.4/lib;/nix/store/4iabmjjq95069myjsrid8pk2ib3yz4nn-openssl-3.0.10/lib;/nix/store/0lvgi2bnwi8cs0wx2242g1275kqx8bba-libssh2-1.11.0/lib;/nix/store/69jpyha5zbll6ppqzhbihhp51lac1hrp-zlib-1.2.13/lib;/nix/store/jwvvq9nyfrjvj10pl533my6d3gpn9nq0-zstd-1.5.5/lib;/nix/store/y2qmx9f9bskxks88l3h6klhzxxwb10ca-curl-8.1.1/lib;/nix/store/ald87b4irf657al0idll1vfggg0m1p6k-procps-3.3.17/lib;/nix/store/c67kb57rfi56aibv5xdb34wgbndb6sdl-util-linux-2.38.1-lib/lib;/nix/store/ammh7161c1qwpvd4ary09vcl246g1giy-nvidia-x11-535.86.05-6.1.45/lib;/nix/store/xfgkyzqz46jxdkrfjz2ksv9cyvaqcx8v-libGL-1.6.0/lib;/nix/store/g1kflwfc5ym6xi59c16qscpr1ibhnyf1-libglvnd-1.6.0/lib;/nix/store/ri6dskyaxr5y3b77mzq9d06iq8rr631a-glu-9.0.2/lib;/nix/store/c751zxvc7l3lp6y792g7mxk1k5ikpbyi-libXfixes-6.0.0/lib;/nix/store/n1iy4vr7ikx2h9pi1ikmh61ayxl8i3g9-libXau-1.0.9/lib;/nix/store/4zkhmhn9krccx97d2ad0z7fna278ksiv-libXext-1.3.4/lib;/nix/store/0sylfkpwq0wvl5whrh0sh9ax7iwg3922-libXi-1.8/lib;/nix/store/jyb9nwgpc8y6k3z4x5nn6bh3r495mw26-libxcb-1.14/lib;/nix/store/z5dlm8l0yzh4d3l2370lb1m2hfmcdfiy-libX11-1.8.6/lib;/nix/store/7bn76v7x92aay575vk0j1rswiza4885x-libICE-1.0.10/lib;/nix/store/q216bjzidc0iivnh86iw76pwbj55ilxq-libSM-1.2.3/lib;/nix/store/ijxbda4kp4mkffz7v7pkb69narm9p0dd-libXt-1.2.1/lib;/nix/store/ngipr3yi3ahk5zhhfivc6r6ybnw6qymg-libXmu-1.1.3/lib;/nix/store/13ykj3314cc6mwzw772id4ghp0jk7lzy-freeglut-3.4.0/lib;/nix/store/qp365vl8w5dvjrk7j118ivga58pvdhs2-libXv-1.0.11/lib;/nix/store/pcj0rysjwcshrw9nwgvii19123a0cs1p-libXrender-0.9.10/lib;/nix/store/nfki84ggsc95inqkj0jfjaxrdkx00jz9-libXrandr-1.5.2/lib;/nix/store/23rqq5igphh0li64kzxw34hr6l713hbv-libpng-apng-1.6.39/lib;/nix/store/g6n75l18fhmlaagq08pwl1lyhvsnzk3z-tbb-2020.3/lib;/nix/store/19s69x56fidcyscaqm8sdf87hswaz8rg-gbenchmark-1.7.1/lib;/nix/store/kpvkzy9v1b8l4f8nvcxsk260lh4vjsz6-ncurses-abi5-compat-6.4/lib;/nix/store/4a6iv3pl9npf4iwm72dskwl89hckdcdj-libffi-3.4.4/lib;/nix/store/fgr89gfh5if2x12qm0wz2s996jyn5qx9-gettext-0.21/lib;/nix/store/4vrk6zldfblhry2hi4p0jsy4j7nsvgaz-glib-2.76.2/lib;/nix/store/vcma91m3khc0snc8wwhghgg1vq2dhm4r-gstreamer-1.22.5/lib;/nix/store/4ykjj160h99v54g5wi2lj28b7985l2z3-gst-plugins-base-1.22.5/lib;/nix/store/mq8l9r7bj2597hqwxkibgpbcm8cr9x2d-gst-plugins-bad-1.22.5/lib;/nix/store/46m4xx889wlhsdj72j38fnlyyvvvvbyb-glibc-2.37-8/lib;/nix/store/pfx4gg15nllsa6cwfhjnink1jr35dpfq-gcc-11.3.0-lib/lib;/nix/store/sw9kw1fz9h7h0pr2hk7jyx8m1ac1ydsg-gcc-wrapper-11.3.0/bin;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/lib/gcc/x86_64-unknown-linux-gnu/11.3.0;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/lib64;/nix/store/3alwf0rzfi7ffha708bsq634znnchkk7-gcc-11.3.0/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/nix/store/lcf37pgp3rgww67v9x2990hbfwx96c1w-gcc-wrapper-12.2.0/bin/ld")
set(CMAKE_AR "/nix/store/lcf37pgp3rgww67v9x2990hbfwx96c1w-gcc-wrapper-12.2.0/bin/ar")
set(CMAKE_MT "")
