import Image from "next/image";
import Link from "next/link";
import cn from "classnames";
import ROUTES from "@/lib/routes";

const Logo = (props: { classNames: string }) => {
  return (
    <Link href={ROUTES.HOME} className={cn("inline-flex", props.classNames)}>
      <span className="relative h-10 md:h-14 w-20 sm:w-32 overflow-hidden md:w-40">
        <Image
          aria-hidden
          src="/logo.svg"
          alt="Formica"
          width={100}
          height={40}
        />
      </span>
    </Link>
  );
};

export default Logo;
