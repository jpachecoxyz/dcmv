{pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {

  packages = [
    pkgs.python312
    (pkgs.python312Packages.opencv4.override {enableGtk3 = true;})
    # opencvWithQt
    pkgs.stdenv.cc.cc
    pkgs.python3Packages.numpy
    pkgs.python3Packages.ultralytics
    pkgs.python3Packages.imutils
    pkgs.cmake
    pkgs.gtk3
  ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libGL
    pkgs.glib
  ];

}
