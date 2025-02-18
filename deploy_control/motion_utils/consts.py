net_interface = "eno1"

kPi = 3.141592654
kPi_2 = 1.57079632

JointIndex = dict(
    # Right leg
    kRightHipYaw=8,
    kRightHipRoll=0,
    kRightHipPitch=1,
    kRightKnee=2,
    kRightAnkle=11,
    # Left leg
    kLeftHipYaw=7,
    kLeftHipRoll=3,
    kLeftHipPitch=4,
    kLeftKnee=5,
    kLeftAnkle=10,
    kWaistYaw=6,
    kNotUsedJoint=9,
    # Right arm
    kRightShoulderPitch=12,
    kRightShoulderRoll=13,
    kRightShoulderYaw=14,
    kRightElbow=15,
    # Left arm
    kLeftShoulderPitch=16,
    kLeftShoulderRoll=17,
    kLeftShoulderYaw=18,
    kLeftElbow=19,
)

SimJointIndex = dict(
    # Right arm
    kRightShoulderPitch=15,
    kRightShoulderRoll=16,
    kRightShoulderYaw=17,
    kRightElbow=18,
    # Left arm
    kLeftShoulderPitch=11,
    kLeftShoulderRoll=12,
    kLeftShoulderYaw=13,
    kLeftElbow=14,
)

ArmJoints = [
    JointIndex["kLeftShoulderPitch"],
    JointIndex["kLeftShoulderRoll"],
    JointIndex["kLeftShoulderYaw"],
    JointIndex["kLeftElbow"],
    JointIndex["kRightShoulderPitch"],
    JointIndex["kRightShoulderRoll"],
    JointIndex["kRightShoulderYaw"],
    JointIndex["kRightElbow"],
]

ArmJointsName = [
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
]


SimArmJointLowerBounds = {
    SimJointIndex["kLeftShoulderPitch"]: -2.5,
    SimJointIndex["kLeftShoulderRoll"]: -0.3,
    SimJointIndex["kLeftShoulderYaw"]: -1.3,
    SimJointIndex["kLeftElbow"]: -1.1,
    SimJointIndex["kRightShoulderPitch"]: -2.5,
    SimJointIndex["kRightShoulderRoll"]: -3.1,
    SimJointIndex["kRightShoulderYaw"]: -4.4,
    SimJointIndex["kRightElbow"]: -1.1,
}

ArmJointLowerBounds = {
    JointIndex["kLeftShoulderPitch"]: -2.5,
    JointIndex["kLeftShoulderRoll"]: -0.3,
    JointIndex["kLeftShoulderYaw"]: -1.3,
    JointIndex["kLeftElbow"]: -1.1,
    JointIndex["kRightShoulderPitch"]: -2.5,
    JointIndex["kRightShoulderRoll"]: -3.1,
    JointIndex["kRightShoulderYaw"]: -4.4,
    JointIndex["kRightElbow"]: -1.1,
}

SimArmJointUpperBounds = {
    SimJointIndex["kLeftShoulderPitch"]: 1.8,
    SimJointIndex["kLeftShoulderRoll"]: 3.1,
    SimJointIndex["kLeftShoulderYaw"]: 4.4,
    SimJointIndex["kLeftElbow"]: kPi_2,
    SimJointIndex["kRightShoulderPitch"]: 1.8,
    SimJointIndex["kRightShoulderRoll"]: 0.3,
    SimJointIndex["kRightShoulderYaw"]: 1.3,
    SimJointIndex["kRightElbow"]: kPi_2,
}

ArmJointUpperBounds = {
    JointIndex["kLeftShoulderPitch"]: 1.8,
    JointIndex["kLeftShoulderRoll"]: 3.1,
    JointIndex["kLeftShoulderYaw"]: 4.4,
    JointIndex["kLeftElbow"]: kPi_2,
    JointIndex["kRightShoulderPitch"]: 1.8,
    JointIndex["kRightShoulderRoll"]: 0.3,
    JointIndex["kRightShoulderYaw"]: 1.3,
    JointIndex["kRightElbow"]: kPi_2,
}

LowerJoints = [
    JointIndex["kLeftHipYaw"],
    JointIndex["kLeftHipRoll"],
    JointIndex["kLeftHipPitch"],
    JointIndex["kLeftKnee"],
    JointIndex["kLeftAnkle"],
    JointIndex["kRightHipYaw"],
    JointIndex["kRightHipRoll"],
    JointIndex["kRightHipPitch"],
    JointIndex["kRightKnee"],
    JointIndex["kRightAnkle"],
    JointIndex["kWaistYaw"],
]

WholeBodyJoints = ArmJoints + LowerJoints

WeakMotors = [
    JointIndex["kLeftAnkle"],
    JointIndex["kRightAnkle"],
    JointIndex["kRightShoulderPitch"],
    JointIndex["kRightShoulderRoll"],
    JointIndex["kRightShoulderYaw"],
    JointIndex["kRightElbow"],
    JointIndex["kLeftShoulderPitch"],
    JointIndex["kLeftShoulderRoll"],
    JointIndex["kLeftShoulderYaw"],
    JointIndex["kLeftElbow"],
]


def is_weak_motor(joint):
    return joint in WeakMotors or joint in ArmJoints


kp_low_ = 60.0
kp_high_ = 200.0
kd_low_ = 1.5
kd_high_ = 5.0

hip_pitch_init_pos_ = -0.5
knee_init_pos_ = 1.0
ankle_init_pos_ = -0.5
shoulder_pitch_init_pos_ = 0.4
