#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "livox_ros_driver2__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__livox_ros_driver2__msg__CustomPoint() -> *const std::ffi::c_void;
}

#[link(name = "livox_ros_driver2__rosidl_generator_c")]
extern "C" {
    fn livox_ros_driver2__msg__CustomPoint__init(msg: *mut CustomPoint) -> bool;
    fn livox_ros_driver2__msg__CustomPoint__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<CustomPoint>, size: usize) -> bool;
    fn livox_ros_driver2__msg__CustomPoint__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<CustomPoint>);
    fn livox_ros_driver2__msg__CustomPoint__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<CustomPoint>, out_seq: *mut rosidl_runtime_rs::Sequence<CustomPoint>) -> bool;
}

// Corresponds to livox_ros_driver2__msg__CustomPoint
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Livox costom pointcloud format.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CustomPoint {
    /// offset time relative to the base time
    pub offset_time: u32,

    /// X axis, unit:m
    pub x: f32,

    /// Y axis, unit:m
    pub y: f32,

    /// Z axis, unit:m
    pub z: f32,

    /// reflectivity, 0~255
    pub reflectivity: u8,

    /// livox tag
    pub tag: u8,

    /// laser number in lidar
    pub line: u8,

}



impl Default for CustomPoint {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !livox_ros_driver2__msg__CustomPoint__init(&mut msg as *mut _) {
        panic!("Call to livox_ros_driver2__msg__CustomPoint__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for CustomPoint {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomPoint__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomPoint__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomPoint__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for CustomPoint {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for CustomPoint where Self: Sized {
  const TYPE_NAME: &'static str = "livox_ros_driver2/msg/CustomPoint";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__livox_ros_driver2__msg__CustomPoint() }
  }
}


#[link(name = "livox_ros_driver2__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__livox_ros_driver2__msg__CustomMsg() -> *const std::ffi::c_void;
}

#[link(name = "livox_ros_driver2__rosidl_generator_c")]
extern "C" {
    fn livox_ros_driver2__msg__CustomMsg__init(msg: *mut CustomMsg) -> bool;
    fn livox_ros_driver2__msg__CustomMsg__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<CustomMsg>, size: usize) -> bool;
    fn livox_ros_driver2__msg__CustomMsg__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<CustomMsg>);
    fn livox_ros_driver2__msg__CustomMsg__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<CustomMsg>, out_seq: *mut rosidl_runtime_rs::Sequence<CustomMsg>) -> bool;
}

// Corresponds to livox_ros_driver2__msg__CustomMsg
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Livox publish pointcloud msg format.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CustomMsg {
    /// ROS standard message header
    pub header: std_msgs::msg::rmw::Header,

    /// The time of first point
    pub timebase: u64,

    /// Total number of pointclouds
    pub point_num: u32,

    /// Lidar device id number
    pub lidar_id: u8,

    /// Reserved use
    pub rsvd: [u8; 3],

    /// Pointcloud data
    pub points: rosidl_runtime_rs::Sequence<super::super::msg::rmw::CustomPoint>,

}



impl Default for CustomMsg {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !livox_ros_driver2__msg__CustomMsg__init(&mut msg as *mut _) {
        panic!("Call to livox_ros_driver2__msg__CustomMsg__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for CustomMsg {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomMsg__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomMsg__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { livox_ros_driver2__msg__CustomMsg__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for CustomMsg {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for CustomMsg where Self: Sized {
  const TYPE_NAME: &'static str = "livox_ros_driver2/msg/CustomMsg";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__livox_ros_driver2__msg__CustomMsg() }
  }
}


